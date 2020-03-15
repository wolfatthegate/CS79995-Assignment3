#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <algorithm>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

#define SIZE 1000
#define N 10
#define RANDMAX 65536

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CHECK(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void computeMovingAverage(float *dev_a, float *dev_b, int size, int n) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x; // 0, 1, 2 
	int halfway = size/2; // 1000/2
	int i = halfway; 
	
	do{
		if(idx < i)
  			if(dev_a[idx+i] > dev_a[idx])
  				dev_a[idx] = dev_a[idx+i]; 
  		 
  		__syncthreads(); 

  		if(i%2==0) 
  			i = i/2; 
  		else 
  			i = (i+1)/2; 

	} while (i > 1);
	
	__syncthreads(); 

	if(dev_a[1]>dev_a[0])
		dev_a[0] = dev_a[1];
	
}

void computeMovingAverageOnCPU(vector<float> &host_a, float &cpuRef, const int size) {	

	int maximum = host_a[0]; 

	for(int i = 0; i < size; i++){

		if(host_a[i] >= maximum){
			maximum = host_a[i]; 		
		}

	}
	cpuRef = maximum; 
	
}

int main(void){

	// set up device
	int dev = 0; 
	cudaDeviceProp deviceProp; 
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev)); 

	int n = N; 
	int size = SIZE; 
	int randmax = RANDMAX;

	printf("Array Size: %d  Sample Size: %d\n", size, N);
	size_t nBytes = size * sizeof(float); 
	float cpuRef = 0.0f; 

	// initialize random number
	srand ((int)time(0));
 
	// initialize vector and generate random indices between 0 and 5. 
	vector<float> host_a(size);
	vector<float> host_b(size); 
	printf("Generating %d random integers from 0 to %d\n", size, randmax); 
	generate(host_a.begin(), host_a.end(), []() { return rand() % RANDMAX; }); 

	float *dev_a, *dev_b; 
	cudaMalloc(&dev_a, nBytes); 
	cudaMalloc(&dev_b, nBytes); 
	cudaMemcpy(dev_a, host_a.data(), nBytes, cudaMemcpyHostToDevice); 
	cudaMemcpy(dev_b, host_b.data(), nBytes, cudaMemcpyHostToDevice); 
	// declare block and grid dimension. 

	dim3 block (size/n); 
	dim3 grid (n); 

	// Timer starts 
	float GPUtime, CPUtime; 
	cudaEvent_t start, stop; 

	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0); 

	printf("Launching Kernel \n"); 
	computeMovingAverage <<< grid, block >>> (dev_a, dev_b, size, n); 
	
	cudaMemcpy(host_a.data(), dev_a, nBytes, cudaMemcpyDeviceToHost); 

	// timer stops
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&GPUtime, start, stop); 

	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0); 

	computeMovingAverageOnCPU(host_a, cpuRef, size);

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&CPUtime, start, stop); 

    printf("Kernel: computeMovingAverage <<<gridDim: %d, blockDim: %d>>>\n", grid.x, block.x); 

	printf("Compute time on GPU: %3.6f ms \n", GPUtime); 
	printf("Compute time on CPU: %3.6f ms \n", CPUtime); 
	printf("Maximum integer found on CPU: %d\n", (int)cpuRef); 
	printf("Maximum integer found on GPU: %d\n", (int)host_a[0]); 

	cudaFree(dev_a);
	cudaFree(dev_b); 

	return (0); 
}