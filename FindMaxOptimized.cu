// FindMaxOptimized.cu
//
// Optimized GPU maximum reduction. This is a rewrite of FindMax.cu that keeps
// the same program shape (fill a random array, find the maximum on the GPU and
// on the CPU, print both results and both run times) but fixes the cross-block
// race in the original kernel and uses the standard fast reduction pattern:
//
//   1. Each thread walks the input with a grid-stride loop and keeps a running
//      maximum in a register. This lets a fixed number of blocks handle an
//      array of any size, and the per-element work is a single coalesced load.
//   2. Each warp reduces its 32 running maxima with __shfl_down_sync, which
//      needs no shared memory and no __syncthreads().
//   3. Lane 0 of each warp writes its result to shared memory. After one
//      __syncthreads(), the first warp reduces those values and writes one
//      maximum per block to global memory.
//   4. A second launch of the same kernel with a single block reduces the
//      per-block results to the final answer. Only threads inside a block ever
//      share data within a launch, so the result is always correct.
//
// The input array is never modified, so the CPU reference reads the original
// data. Kernel time is measured separately from the copy back to the host.

#include <cuda_runtime.h>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <vector>

#define SIZE 1000          // default number of elements; override with argv[1]
#define RANDMAX 65536      // random values are in [0, RANDMAX)
#define BLOCK_SIZE 256     // threads per block; must be a multiple of 32 and <= 1024
#define MAX_BLOCKS 1024    // cap on the first-stage grid; the grid-stride loop covers the rest

#define CHECK(call)                                                           \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Reduce 32 values held by the lanes of one warp down to lane 0.
__device__ __forceinline__ float warpReduceMax(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    return v;
}

// out[blockIdx.x] = max(in[0 .. n-1]) over the elements this block visits.
template <int BLOCK>
__global__ void findMaxKernel(const float* __restrict__ in,
                              float* __restrict__ out, int n) {
    static_assert(BLOCK % 32 == 0 && BLOCK <= 1024,
                  "BLOCK must be a multiple of 32 and at most 1024");
    __shared__ float warpMax[BLOCK / 32];

    // Stage 1: grid-stride loop, running maximum in a register.
    float v = -FLT_MAX;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK)
        v = fmaxf(v, in[i]);

    // Stage 2: reduce within each warp.
    v = warpReduceMax(v);

    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    if (lane == 0) warpMax[warpId] = v;
    __syncthreads();

    // Stage 3: the first warp reduces the per-warp maxima.
    if (warpId == 0) {
        v = (lane < BLOCK / 32) ? warpMax[lane] : -FLT_MAX;
        v = warpReduceMax(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

// Launch the two-stage reduction. dev_partial must hold at least numBlocks
// floats; the final answer is written to dev_result[0].
static void findMaxOnGPU(const float* dev_in, float* dev_partial,
                         float* dev_result, int n, int numBlocks) {
    findMaxKernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE>>>(dev_in, dev_partial, n);
    findMaxKernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(dev_partial, dev_result, numBlocks);
}

static float findMaxOnCPU(const std::vector<float>& a) {
    float m = -FLT_MAX;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] > m) m = a[i];
    return m;
}

int main(int argc, char** argv) {
    int size = (argc > 1) ? atoi(argv[1]) : SIZE;
    if (size <= 0) {
        fprintf(stderr, "Array size must be positive\n");
        return EXIT_FAILURE;
    }

    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    CHECK(cudaSetDevice(dev));
    printf("Using Device %d: %s\n", dev, prop.name);

    const int numBlocks = std::min(MAX_BLOCKS, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const size_t nBytes = (size_t)size * sizeof(float);

    printf("Array Size: %d  Blocks: %d  Threads per block: %d\n",
           size, numBlocks, BLOCK_SIZE);
    printf("Generating %d random integers from 0 to %d\n", size, RANDMAX);

    srand((unsigned)time(0));
    std::vector<float> host_a(size);
    for (int i = 0; i < size; ++i) host_a[i] = (float)(rand() % RANDMAX);

    float *dev_a = nullptr, *dev_partial = nullptr, *dev_result = nullptr;
    CHECK(cudaMalloc(&dev_a, nBytes));
    CHECK(cudaMalloc(&dev_partial, numBlocks * sizeof(float)));
    CHECK(cudaMalloc(&dev_result, sizeof(float)));
    CHECK(cudaMemcpy(dev_a, host_a.data(), nBytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warm-up launch so the timed run does not include one-time module load.
    findMaxOnGPU(dev_a, dev_partial, dev_result, size, numBlocks);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Timed run: kernels only.
    printf("Launching Kernel\n");
    CHECK(cudaEventRecord(start, 0));
    findMaxOnGPU(dev_a, dev_partial, dev_result, size, numBlocks);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaGetLastError());
    float kernelTime = 0.0f;
    CHECK(cudaEventElapsedTime(&kernelTime, start, stop));

    // Copy back the single result (4 bytes) and time that separately.
    float gpuResult = 0.0f;
    CHECK(cudaEventRecord(start, 0));
    CHECK(cudaMemcpy(&gpuResult, dev_result, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float copyTime = 0.0f;
    CHECK(cudaEventElapsedTime(&copyTime, start, stop));

    // CPU reference on the untouched input.
    auto cpuStart = std::chrono::high_resolution_clock::now();
    float cpuResult = findMaxOnCPU(host_a);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float, std::milli>(cpuStop - cpuStart).count();

    printf("Kernel: findMaxKernel <<<gridDim: %d, blockDim: %d>>> then <<<1, %d>>>\n",
           numBlocks, BLOCK_SIZE, BLOCK_SIZE);
    printf("Kernel time on GPU:        %3.6f ms\n", kernelTime);
    printf("Result copy to host:       %3.6f ms\n", copyTime);
    printf("Total GPU (kernel + copy): %3.6f ms\n", kernelTime + copyTime);
    printf("Compute time on CPU:       %3.6f ms\n", cpuTime);
    printf("Maximum integer found on CPU: %d\n", (int)cpuResult);
    printf("Maximum integer found on GPU: %d\n", (int)gpuResult);
    printf("Results %s\n", (cpuResult == gpuResult) ? "match" : "DO NOT MATCH");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_partial));
    CHECK(cudaFree(dev_result));

    return (cpuResult == gpuResult) ? EXIT_SUCCESS : EXIT_FAILURE;
}
