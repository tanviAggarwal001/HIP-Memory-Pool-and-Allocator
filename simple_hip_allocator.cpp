// Compile: hipcc -std=c++17 simple_hip_allocator.cpp -O2 -o simple_hip_allocator

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cassert>
#include <random>

#define HIP_CALL(call)                                                      \
    do {                                                                    \
        hipError_t err = (call);                                            \
        if (err != hipSuccess) {                                            \
            std::cerr << "HIP error " << hipGetErrorString(err)             \
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
class SimpleHipAllocator {
public:
    struct FreeBlock {
        size_t offset;
        size_t size;
        FreeBlock* next;
    };

    SimpleHipAllocator(size_t totalBytes, size_t alignment = 256)
        : totalSize_(align_up(totalBytes, alignment)), alignment_(alignment),
          freeList_(nullptr), basePtr_(nullptr)
    {
        HIP_CALL(hipMalloc(&basePtr_, totalSize_));
        // start with a single free block
        freeList_ = new FreeBlock{0, totalSize_, nullptr};
    }

    ~SimpleHipAllocator() {
        if (basePtr_) HIP_CALL(hipFree(basePtr_));
        std::lock_guard<std::mutex> g(mutex_);
        FreeBlock* cur = freeList_;
        while (cur) {
            FreeBlock* nxt = cur->next;
            delete cur;
            cur = nxt;
        }
    }

    // allocate 'size' bytes, returns device pointer (void*)
    void* alloc(size_t reqSize) {
        size_t size = align_up(reqSize, alignment_);
        std::lock_guard<std::mutex> g(mutex_);
        FreeBlock **p = &freeList_;
        while (*p) {
            if ((*p)->size >= size) {
                size_t allocOffset = (*p)->offset;
                if ((*p)->size > size + minSplitSize_) {
                    // split block: allocate bottom portion
                    (*p)->offset += size;
                    (*p)->size -= size;
                } else {
                    // use whole block
                    FreeBlock* toDelete = *p;
                    *p = (*p)->next;
                    delete toDelete;
                }
                return static_cast<char*>(basePtr_) + allocOffset;
            }
            p = &((*p)->next);
        }
        // no block large enough
        return nullptr;
    }

    // free pointer (and size). Must pass size that was allocated (or store it externally).
    void free(void* devPtr, size_t reqSize) {
        if (!devPtr) return;
        size_t size = align_up(reqSize, alignment_);
        size_t offset = static_cast<char*>(devPtr) - static_cast<char*>(basePtr_);
        std::lock_guard<std::mutex> g(mutex_);

        // insert and coalesce; keep free list sorted by offset
        FreeBlock* prev = nullptr;
        FreeBlock* cur = freeList_;
        while (cur && cur->offset < offset) {
            prev = cur;
            cur = cur->next;
        }
        // try to coalesce with prev and cur
        if (prev && prev->offset + prev->size == offset) {
            // merge into prev
            prev->size += size;
            // also merge with cur if adjacent
            if (cur && offset + size == cur->offset) {
                prev->size += cur->size;
                prev->next = cur->next;
                delete cur;
            }
        } else if (cur && offset + size == cur->offset) {
            // merge with cur at front
            cur->offset = offset;
            cur->size += size;
        } else {
            // insert new block
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
// Device-side bump allocator (no free)
// -----------------------------
struct DeviceAllocatorArgs {
    unsigned char* base;             // device pointer base
    unsigned long long* counter;     // device pointer to 64-bit counter
    size_t capacity;
};

__device__ inline void* device_alloc(DeviceAllocatorArgs args, size_t bytes) {
    const size_t align = 8;
    size_t s = (bytes + (align - 1)) & ~(align - 1);
    // atomicAdd for unsigned long long
    unsigned long long old = atomicAdd(args.counter, (unsigned long long)s);
    if (old + s > args.capacity) return nullptr;
    return (void*)(args.base + old);
}

__global__ void example_kernel_alloc(DeviceAllocatorArgs args, int n, int reqSize, void** outPtrs) {
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
double benchmark_host_allocator(SimpleHipAllocator &alloc, int N, int sizePerAlloc) {
    std::vector<void*> ptrs(N, nullptr);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        ptrs[i] = alloc.alloc(sizePerAlloc);
        if (!ptrs[i]) {
            std::cerr << "Allocator out of memory during benchmark\n";
            break;
        }
    }
    for (int i = 0; i < N; ++i) {
        alloc.free(ptrs[i], sizePerAlloc);
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double benchmark_hip_malloc(int N, int sizePerAlloc) {
    std::vector<void*> ptrs(N, nullptr);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        hipError_t e = hipMalloc(&ptrs[i], sizePerAlloc);
        if (e != hipSuccess) {
            std::cerr << "hipMalloc failed during benchmark: " << hipGetErrorString(e) << "\n";
            break;
        }
    }
    for (int i = 0; i < N; ++i) {
        if (ptrs[i]) hipFree(ptrs[i]);
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// -----------------------------
// Example usage
// -----------------------------
int main() {
    const size_t poolSize = size_t(128) * 1024 * 1024; // 128 MB pool
    SimpleHipAllocator allocator(poolSize);

    std::cout << "Pool size: " << allocator.total_size() / (1024*1024) << " MB\n";

    // Benchmark host allocator vs hipMalloc for many small allocations
    int N = 10000;
    int smallSize = 256; // 256 bytes each

    std::cout << "Running benchmark: " << N << " allocations of " << smallSize << " bytes\n";
    double t_alloc = benchmark_host_allocator(allocator, N, smallSize);
    std::cout << "SimpleHipAllocator time (ms): " << t_alloc << "\n";

    double t_hipm = benchmark_hip_malloc(N, smallSize);
    std::cout << "hipMalloc + hipFree time (ms): " << t_hipm << "\n";

    // Device-side allocator example
    unsigned char* d_pool = nullptr;
    HIP_CALL(hipMalloc(&d_pool, 4 * 1024 * 1024)); // 4 MB device pool
    unsigned long long* d_counter = nullptr;
    HIP_CALL(hipMalloc(&d_counter, sizeof(unsigned long long)));
    HIP_CALL(hipMemset(d_counter, 0, sizeof(unsigned long long)));

    DeviceAllocatorArgs args{};
    args.base = d_pool;
    args.counter = d_counter;
    args.capacity = 4 * 1024 * 1024;

    // prepare output pointers array
    int threads = 1024;
    int blocks = (threads + 255) / 256;
    int totalThreads = blocks * 256;
    void** d_out = nullptr;
    HIP_CALL(hipMalloc(&d_out, totalThreads * sizeof(void*)));
    HIP_CALL(hipMemset(d_out, 0, totalThreads * sizeof(void*)));

    // Launch kernel: each thread allocates 64 bytes on device allocator
    int req = 64;
    dim3 grid(blocks);
    dim3 block(256);
    hipLaunchKernelGGL(example_kernel_alloc, grid, block, 0, 0, args, totalThreads, req, d_out);
    HIP_CALL(hipDeviceSynchronize());

    // Copy-back a few pointers to verify
    std::vector<void*> sample(totalThreads);
    HIP_CALL(hipMemcpy(sample.data(), d_out, totalThreads * sizeof(void*), hipMemcpyDeviceToHost));
    int nonNull = 0;
    for (int i = 0; i < totalThreads; ++i) if (sample[i]) ++nonNull;
    std::cout << "device allocator: " << nonNull << " allocations succeeded out of " << totalThreads << "\n";

    // cleanup
    HIP_CALL(hipFree(d_out));
    HIP_CALL(hipFree(d_pool));
    HIP_CALL(hipFree(d_counter));

    HIP_CALL(hipDeviceReset());
    return 0;
}
