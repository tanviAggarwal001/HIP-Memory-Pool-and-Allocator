// Compile: nvcc -std=c++17 simple_cuda_allocator.cu -O2 -o simple_cuda_allocator

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cassert>
#include <random>

#define CUDA_CALL(call)                                                     \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

static inline size_t align_up(size_t v, size_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

// -----------------------------
// Host-side simple allocator
// -----------------------------
class SimpleCudaAllocator {
public:
    struct FreeBlock {
        size_t offset;
        size_t size;
        FreeBlock* next;
    };

    SimpleCudaAllocator(size_t totalBytes, size_t alignment = 256)
        : totalSize_(align_up(totalBytes, alignment)),
          alignment_(alignment),
          freeList_(nullptr),
          basePtr_(nullptr)
    {
        CUDA_CALL(cudaMalloc(&basePtr_, totalSize_));
        freeList_ = new FreeBlock{0, totalSize_, nullptr};
    }

    ~SimpleCudaAllocator() {
        if (basePtr_) CUDA_CALL(cudaFree(basePtr_));
        std::lock_guard<std::mutex> g(mutex_);
        FreeBlock* cur = freeList_;
        while (cur) {
            FreeBlock* nxt = cur->next;
            delete cur;
            cur = nxt;
        }
    }

    void* alloc(size_t reqSize) {
        size_t size = align_up(reqSize, alignment_);
        std::lock_guard<std::mutex> g(mutex_);
        FreeBlock** p = &freeList_;
        while (*p) {
            if ((*p)->size >= size) {
                size_t allocOffset = (*p)->offset;
                if ((*p)->size > size + minSplitSize_) {
                    (*p)->offset += size;
                    (*p)->size -= size;
                } else {
                    FreeBlock* toDelete = *p;
                    *p = (*p)->next;
                    delete toDelete;
                }
                return static_cast<char*>(basePtr_) + allocOffset;
            }
            p = &((*p)->next);
        }
        return nullptr;
    }

    void free(void* devPtr, size_t reqSize) {
        if (!devPtr) return;
        size_t size = align_up(reqSize, alignment_);
        size_t offset =
            static_cast<char*>(devPtr) - static_cast<char*>(basePtr_);

        std::lock_guard<std::mutex> g(mutex_);
        FreeBlock* prev = nullptr;
        FreeBlock* cur = freeList_;

        while (cur && cur->offset < offset) {
            prev = cur;
            cur = cur->next;
        }

        if (prev && prev->offset + prev->size == offset) {
            prev->size += size;
            if (cur && offset + size == cur->offset) {
                prev->size += cur->size;
                prev->next = cur->next;
                delete cur;
            }
        } else if (cur && offset + size == cur->offset) {
            cur->offset = offset;
            cur->size += size;
        } else {
            FreeBlock* nb = new FreeBlock{offset, size, cur};
            if (prev) prev->next = nb;
            else freeList_ = nb;
        }
    }

    void* base_ptr() const { return basePtr_; }
    size_t total_size() const { return totalSize_; }

private:
    void* basePtr_;
    size_t totalSize_;
    size_t alignment_;
    FreeBlock* freeList_;
    std::mutex mutex_;
    const size_t minSplitSize_ = 64;
};

// -----------------------------
// Device-side bump allocator
// -----------------------------
struct DeviceAllocatorArgs {
    unsigned char* base;
    unsigned long long* counter;
    size_t capacity;
};

__device__ inline void* device_alloc(DeviceAllocatorArgs args, size_t bytes) {
    const size_t align = 8;
    size_t s = (bytes + (align - 1)) & ~(align - 1);
    unsigned long long old =
        atomicAdd(args.counter, (unsigned long long)s);
    if (old + s > args.capacity) return nullptr;
    return (void*)(args.base + old);
}

__global__ void example_kernel_alloc(DeviceAllocatorArgs args,
                                     int n,
                                     int reqSize,
                                     void** outPtrs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    void* p = device_alloc(args, reqSize);
    outPtrs[tid] = p;
    if (p) {
        unsigned char* q = (unsigned char*)p;
        q[0] = (unsigned char)(tid & 0xFF);
    }
}

// -----------------------------
// Benchmark: host allocations
// -----------------------------
double benchmark_host_allocator(SimpleCudaAllocator& alloc,
                                int N,
                                int sizePerAlloc) {
    std::vector<void*> ptrs(N, nullptr);
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < N; ++i) {
        ptrs[i] = alloc.alloc(sizePerAlloc);
        if (!ptrs[i]) {
            std::cerr << "Allocator out of memory\n";
            break;
        }
    }

    for (int i = 0; i < N; ++i) {
        alloc.free(ptrs[i], sizePerAlloc);
    }

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double benchmark_cuda_malloc(int N, int sizePerAlloc) {
    std::vector<void*> ptrs(N, nullptr);
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < N; ++i) {
        cudaError_t e = cudaMalloc(&ptrs[i], sizePerAlloc);
        if (e != cudaSuccess) {
            std::cerr << "cudaMalloc failed: "
                      << cudaGetErrorString(e) << "\n";
            break;
        }
    }

    for (int i = 0; i < N; ++i) {
        if (ptrs[i]) cudaFree(ptrs[i]);
    }

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -----------------------------
// Example usage
// -----------------------------
int main() {
    const size_t poolSize = size_t(128) * 1024 * 1024;
    SimpleCudaAllocator allocator(poolSize);

    std::cout << "Pool size: "
              << allocator.total_size() / (1024 * 1024)
              << " MB\n";

    int N = 10000;
    int smallSize = 256;

    std::cout << "Running benchmark: " << N
              << " allocations of " << smallSize << " bytes\n";

    double t_alloc = benchmark_host_allocator(allocator, N, smallSize);
    std::cout << "SimpleCudaAllocator time (ms): "
              << t_alloc << "\n";

    double t_cuda = benchmark_cuda_malloc(N, smallSize);
    std::cout << "cudaMalloc + cudaFree time (ms): "
              << t_cuda << "\n";

    // Device-side allocator example
    unsigned char* d_pool = nullptr;
    CUDA_CALL(cudaMalloc(&d_pool, 4 * 1024 * 1024));

    unsigned long long* d_counter = nullptr;
    CUDA_CALL(cudaMalloc(&d_counter, sizeof(unsigned long long)));
    CUDA_CALL(cudaMemset(d_counter, 0, sizeof(unsigned long long)));

    DeviceAllocatorArgs args{};
    args.base = d_pool;
    args.counter = d_counter;
    args.capacity = 4 * 1024 * 1024;

    int threads = 1024;
    int blocks = (threads + 255) / 256;
    int totalThreads = blocks * 256;

    void** d_out = nullptr;
    CUDA_CALL(cudaMalloc(&d_out, totalThreads * sizeof(void*)));
    CUDA_CALL(cudaMemset(d_out, 0, totalThreads * sizeof(void*)));

    int req = 64;
    example_kernel_alloc<<<blocks, 256>>>(args, totalThreads, req, d_out);
    CUDA_CALL(cudaDeviceSynchronize());

    std::vector<void*> sample(totalThreads);
    CUDA_CALL(cudaMemcpy(sample.data(), d_out,
                         totalThreads * sizeof(void*),
                         cudaMemcpyDeviceToHost));

    int nonNull = 0;
    for (void* p : sample)
        if (p) ++nonNull;

    std::cout << "device allocator: " << nonNull
              << " allocations succeeded out of "
              << totalThreads << "\n";

    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_pool));
    CUDA_CALL(cudaFree(d_counter));

    CUDA_CALL(cudaDeviceReset());
    return 0;
}
