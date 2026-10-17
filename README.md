# CS79995 Assignment 3: Finding the Maximum in CUDA

This repository finds the largest value in an array on the GPU with a parallel reduction, and checks the answer against a sequential loop on the CPU. It contains two versions of the program:

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `FindMax.cu`          | The original submission. Single in-place kernel. Kept unchanged for reference. |
| `FindMaxOptimized.cu` | A rewrite that fixes the cross-block race and uses the standard fast reduction pattern (grid-stride loop, warp shuffles, shared memory, two-stage launch). |

## Requirements

- An NVIDIA GPU with a CUDA-capable driver
- The CUDA Toolkit (`nvcc`)

The optimized version uses `__shfl_down_sync`, which needs compute capability 3.0 or later and CUDA 9 or later.

## Build and run

```bash
# Original
nvcc -std=c++11 -o FindMax FindMax.cu
./FindMax

# Optimized (default 1000 elements, or pass a size on the command line)
nvcc -std=c++11 -O2 -o FindMaxOptimized FindMaxOptimized.cu
./FindMaxOptimized
./FindMaxOptimized 10000000
```

The optimized program exits with status 0 when the CPU and GPU results match and 1 when they do not, so it can be used as a self-check in a script.

---

## `FindMax.cu` (original)

### How it works

1. The host fills an array of `SIZE` floats with random integers in `[0, RANDMAX)`.
2. The array is copied to the device, and the `computeMovingAverage` kernel runs on `N` blocks of `SIZE / N` threads each.
3. The kernel reduces the array in place. On each pass, thread `idx` compares `a[idx]` with `a[idx + i]` and keeps the larger value. The stride `i` starts at `SIZE / 2` and is halved (rounded up for odd values) until it reaches 1. A last comparison of `a[0]` and `a[1]` leaves the maximum in `a[0]`.
4. The result is copied back to the host, and a linear scan on the CPU (`computeMovingAverageOnCPU`) gives a reference value.
5. Both run times are measured with CUDA events and printed along with both results.

The kernel and CPU function names come from an earlier assignment. They compute a maximum, not a moving average.

### Example output

```
Using Device 0: <your GPU name>
Array Size: 1000  Sample Size: 10
Generating 1000 random integers from 0 to 65536
Launching Kernel
Kernel: computeMovingAverage <<<gridDim: 10, blockDim: 100>>>
Compute time on GPU: 0.123456 ms
Compute time on CPU: 0.012345 ms
Maximum integer found on CPU: 65501
Maximum integer found on GPU: 65501
```

### Configuration

| Macro     | Default | Meaning                                             |
|-----------|---------|-----------------------------------------------------|
| `SIZE`    | 1000    | Number of elements in the array                     |
| `N`       | 10      | Number of blocks; each block has `SIZE / N` threads |
| `RANDMAX` | 65536   | Upper bound (exclusive) of the random values        |

`SIZE` should divide evenly by `N`, and `SIZE / N` must not exceed your GPU's maximum threads per block (usually 1024).

### Known limitations

These are the problems the optimized version was written to address.

- **Cross-block race.** `__syncthreads()` only synchronizes threads within one block. The kernel runs on several blocks, so a thread in one block can read `a[idx + i]` before the thread in another block that owns that element has finished updating it. The GPU result can then be smaller than the true maximum. The bug is intermittent, which makes it easy to miss in testing.
- **GPU time includes the copy back.** The timed region covers the kernel plus a `cudaMemcpy` of the whole array to the host, so the reported "GPU time" is mostly transfer time rather than compute time. It also includes the one-time cost of loading the kernel on first launch.
- **The CPU reference reads modified data.** The CPU scan runs on the array after it has been copied back from the device, not on the original input. It still finds the right maximum because the reduction only overwrites values with larger values already present, but it is not an independent check.
- **Every element is read many times.** Each pass of the loop re-reads from global memory. For `SIZE` elements the kernel makes about `log2(SIZE)` passes, each a round trip to global memory, and half of the threads are idle on every pass.
- **Rigid launch geometry.** `SIZE / N` threads per block means a large `SIZE` needs a large `N` by hand, and the array size must be a multiple of `N`.
- **Unchecked calls and unused buffers.** `cudaMalloc`, `cudaMemcpy`, and the kernel launch are not checked for errors. `dev_b` and `host_b` are allocated but never used.

---

## `FindMaxOptimized.cu`

### How it works

The optimized kernel, `findMaxKernel`, never shares data between blocks inside a single launch. It uses four stages:

1. **Grid-stride loop.** Each thread starts at its global index and steps forward by the total number of threads in the grid, keeping a running maximum in a register. This makes every load coalesced, lets a fixed number of blocks cover an array of any size, and means each element is read from global memory exactly once.
2. **Warp-level reduction.** The 32 threads in a warp combine their running maxima with `__shfl_down_sync`. Shuffles move data directly between registers, so this stage needs no shared memory and no `__syncthreads()`.
3. **Block-level reduction.** Lane 0 of each warp writes its result to a small shared-memory array (one float per warp, 8 floats for a 256-thread block). After one `__syncthreads()`, the first warp reduces those values with another round of shuffles and writes one maximum per block to global memory.
4. **Second launch.** The same kernel runs again with a single block over the per-block results. Because a kernel launch is a grid-wide barrier, every per-block maximum is complete before this launch starts, so the final result is always correct.

The input array is read-only on the device (`const float* __restrict__`), so the CPU reference scans the original data and is a genuinely independent check.

Timing is split into three numbers: kernel time alone, the time to copy the single 4-byte result back, and their sum. A warm-up launch runs before the timed launch so that one-time module loading does not inflate the kernel time. CPU time is measured with `std::chrono`.

### Example output

```
Using Device 0: <your GPU name>
Array Size: 1000  Blocks: 4  Threads per block: 256
Generating 1000 random integers from 0 to 65536
Launching Kernel
Kernel: findMaxKernel <<<gridDim: 4, blockDim: 256>>> then <<<1, 256>>>
Kernel time on GPU:        0.012345 ms
Result copy to host:       0.010000 ms
Total GPU (kernel + copy): 0.022345 ms
Compute time on CPU:       0.001234 ms
Maximum integer found on CPU: 65501
Maximum integer found on GPU: 65501
Results match
```

At 1000 elements the CPU will usually win, because the launch overhead of two kernels is larger than the time to scan 1000 floats. Run with a size of a few million to see the GPU pull ahead.

### Configuration

| Setting      | Default | Meaning                                                                          |
|--------------|---------|----------------------------------------------------------------------------------|
| `argv[1]`    | `SIZE`  | Array size, given on the command line. Any positive value works.                  |
| `SIZE`       | 1000    | Array size used when no command-line argument is given                           |
| `RANDMAX`    | 65536   | Upper bound (exclusive) of the random values                                     |
| `BLOCK_SIZE` | 256     | Threads per block. Must be a multiple of 32 and at most 1024 (checked at compile time). |
| `MAX_BLOCKS` | 1024    | Cap on the first-stage grid. Larger arrays are covered by the grid-stride loop.   |

There is no divisibility requirement between the array size and the block or grid size.

---

## What changed and why

| Aspect                      | `FindMax.cu` (original)                                                    | `FindMaxOptimized.cu`                                                                  |
|-----------------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Correctness across blocks   | Relies on `__syncthreads()` to order reads and writes between blocks. It does not, so the result is occasionally wrong. | Blocks never share data within a launch. A second launch combines per-block results. Always correct. |
| Global memory traffic       | About `log2(SIZE)` passes, each reading and writing global memory.          | Each input element is read exactly once. Everything after that happens in registers and shared memory. |
| Synchronization             | One `__syncthreads()` per pass, about `log2(SIZE)` per thread.              | One `__syncthreads()` per block in total. Warp stages use shuffles, which need no barrier. |
| Thread utilization          | Half of the active threads go idle on every pass.                           | All threads do useful work in the grid-stride loop; idle lanes only appear in the last two shuffle stages, which are a fixed 10 instructions. |
| Input data                  | Reduced in place, so the input is destroyed.                                | Input is read-only. The CPU reference checks against the original data.                |
| Launch geometry             | `SIZE / N` threads per block. `SIZE` must be a multiple of `N` and `SIZE / N` must be at most 1024. | Fixed 256-thread blocks, grid capped at 1024. Any array size works, set at run time.    |
| GPU timing                  | Kernel plus a full-array copy to the host, including first-launch overhead. | Kernel time, result-copy time (4 bytes), and the sum reported separately, after a warm-up launch. |
| CPU timing                  | CUDA events around a host loop.                                             | `std::chrono` around the host loop.                                                     |
| CPU reference               | Stores the running maximum in an `int` and compares with `>=`.             | Stores the running maximum as a `float` and compares with `>`.                          |
| Error handling              | Only the two device-property calls are checked.                             | Every CUDA call and both kernel launches are checked. Errors print the CUDA message and the line. |
| Unused code                 | `dev_b` and `host_b` are allocated and copied but never read.               | Removed. Only the input array, a small per-block scratch buffer, and a one-float result are allocated. |
| Self-check                  | Prints both values; the user compares them by eye.                          | Prints `Results match` or `Results DO NOT MATCH` and returns a non-zero exit code on mismatch. |

### Why the original race matters

With `SIZE = 1000` and `N = 10`, each block has 100 threads, and the first pass uses stride 500. On that pass thread 250 (in block 2) compares `a[750]` with `a[250]` and may write `a[250]`. On the second pass, stride 250, thread 0 (in block 0) reads `a[250]` and may write `a[0]`. For thread 0 to see the correct value, block 2 must have finished its first pass before block 0 starts its second. Nothing enforces that: the `__syncthreads()` in block 0 only waits for the other 99 threads of block 0, and blocks are scheduled independently. If the true maximum is `a[750]` and block 0 runs ahead of block 2, the GPU reports a smaller value. The two-launch design in the optimized version removes this dependency entirely, because the first launch has fully completed before the second one reads its results.
