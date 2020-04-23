///
/// CUDA lab, AGH course 2020 summer
/// This code is especially prepared by NVIDIA team for studying
/// memory mamagement techniques. We are going to use it as working
/// template for iterative optimisation.
///

#include <stdio.h>
#include <iostream>
#include "timer/timer.h"

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void initGPU(float num, float *a, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for(int i = index; i < N; i += stride)
	{
		a[i] = num;
	}
}

/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/*
 * Host function to confirm values in `vector`. This function
 * assumes all values are the same `target` value.
 */

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  //printf("Success! All values calculated correctly.\n");
}

int main()
{
	int iter = 10;
	double time = 0.0;
	const int N = 2<<24;
	for(int i = 0; i < iter; i++){
		  int deviceId;
		  size_t size = N * sizeof(float);
		  cudaGetDevice(&deviceId);

		  float *a;
		  float *b;
		  float *c;

		  cuda::timer::start();

		  cudaMallocManaged(&a, size);
		  cudaMallocManaged(&b, size);
		  cudaMallocManaged(&c, size);

		  //cudaMemPrefetchAsync(a, size, deviceId);
		  //cudaMemPrefetchAsync(b, size, deviceId);
		  //cudaMemPrefetchAsync(c, size, deviceId);

		  size_t threadsPerBlock;
		  size_t numberOfBlocks;

		  threadsPerBlock = 64;
		  numberOfBlocks = 256;

		  cudaError_t addVectorsErr;
		  cudaError_t asyncErr;

		  initGPU<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
		  initGPU<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
		  initGPU<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);
		  cudaDeviceSynchronize();


		  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

		  addVectorsErr = cudaGetLastError();
		  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

		  asyncErr = cudaDeviceSynchronize();
		  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

		  cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);
		  cudaMemPrefetchAsync(b, size, cudaCpuDeviceId);
		  cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
		  checkElementsAre(3, a, N);
		  checkElementsAre(4, b, N);
		  checkElementsAre(7, c, N);

		  cudaFree(a);
		  cudaFree(b);
		  cudaFree(c);
		  cuda::timer::stop();
		  time += cuda::timer::read();
	}
	std::cout << "[DONE] cpu pref Time: " << time / iter << " s\n";
	time = 0.0;
	for(int i = 0; i < iter; i++){
		  size_t size = N * sizeof(float);

		  float *a;
		  float *b;
		  float *c;

		  cuda::timer::start();

		  cudaMallocManaged(&a, size);
		  cudaMallocManaged(&b, size);
		  cudaMallocManaged(&c, size);



		  size_t threadsPerBlock;
		  size_t numberOfBlocks;

		  /*
		   * nsys should register performance changes when execution configuration
		   * is updated.
		   */

		  threadsPerBlock = 64;
		  numberOfBlocks = 256;

		  cudaError_t addVectorsErr;
		  cudaError_t asyncErr;

		  initGPU<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
		  initGPU<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
		  initGPU<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);
		  cudaDeviceSynchronize();

		  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

		  addVectorsErr = cudaGetLastError();
		  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

		  asyncErr = cudaDeviceSynchronize();
		  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

		  checkElementsAre(7, c, N);
		  checkElementsAre(3, a, N);
		  checkElementsAre(4, b, N);

		  cudaFree(a);
		  cudaFree(b);
		  cudaFree(c);
		  cuda::timer::stop();
		  time += cuda::timer::read();
	}
	std::cout << "[DONE] gpu Time: " << time / iter << " s\n";
}
