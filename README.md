# CS79995 Assignment 3: Finding the Maximum in CUDA

This program finds the largest value in an array on the GPU with a parallel reduction. It also finds the maximum with a sequential loop on the CPU so you can check the answer and compare the two run times.

## How it works

1. The host fills an array of `SIZE` floats with random integers in `[0, RANDMAX)`.
2. The array is copied to the device, and the `computeMovingAverage` kernel runs on `N` blocks of `SIZE / N` threads each.
3. The kernel reduces the array in place. On each pass, thread `idx` compares `a[idx]` with `a[idx + i]` and keeps the larger value. The stride `i` starts at `SIZE / 2` and is halved (rounded up for odd values) until it reaches 1. A last comparison of `a[0]` and `a[1]` leaves the maximum in `a[0]`.
4. The result is copied back to the host, and a linear scan on the CPU (`computeMovingAverageOnCPU`) gives a reference value.
5. Both run times are measured with CUDA events and printed along with both results.

The kernel and CPU function names come from an earlier assignment. They compute a maximum, not a moving average.

## Requirements

- An NVIDIA GPU with a CUDA-capable driver
- The CUDA Toolkit (`nvcc`)

## Build and run

```bash
nvcc -std=c++11 -o FindMax FindMax.cu
./FindMax
```

Example output (the values change on every run):

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

## Configuration

Change these macros at the top of `FindMax.cu`:

| Macro     | Default | Meaning                                          |
|-----------|---------|--------------------------------------------------|
| `SIZE`    | 1000    | Number of elements in the array                  |
| `N`       | 10      | Number of blocks; each block has `SIZE / N` threads |
| `RANDMAX` | 65536   | Upper bound (exclusive) of the random values     |

`SIZE` should divide evenly by `N`, and `SIZE / N` must not exceed your GPU's maximum threads per block (usually 1024).

## Known limitations

- `__syncthreads()` only synchronizes threads within one block. The kernel runs on several blocks, so threads in different blocks can read an element before another block has updated it. The GPU result can then differ from the CPU result. Running a single block, or reducing per block and then combining the block results in a second step, avoids this.
- The GPU time includes copying the result back to the host, so it is not kernel time alone.
- The CPU reference scans the array after it has been copied back from the device, not the original input. It still finds the right maximum, because the reduction only overwrites values with larger values already in the array.
- `dev_b` and `host_b` are allocated but not used.
