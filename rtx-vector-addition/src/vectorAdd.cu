/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

struct timer {
  using timestamp = std::chrono::time_point<std::chrono::system_clock>;
  using duration = std::chrono::duration<double>;

  static void start() { m_timestamp1 = std::chrono::system_clock::now(); }

  static void stop() {
    m_timestamp2 = std::chrono::system_clock::now();
    m_duration = m_timestamp2 - m_timestamp1;
  }

  static double read() { return m_duration.count(); }

  static timestamp m_timestamp1;
  static timestamp m_timestamp2;
  static duration m_duration;
};

timer::timestamp timer::m_timestamp1 = {};
timer::timestamp timer::m_timestamp2 = {};
timer::duration timer::m_duration = {};

void allocateCuda(float** d, size_t size) {
	cudaError_t err = cudaMalloc((void **)d, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void memcopyCudaHostToDevice(void* host, void* device, size_t size) {
	cudaError_t err = cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void memcopyCudaDeviceToHost(void* host, void* device, size_t size) {
	cudaError_t err = cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void freeCuda(void *ptr) {
	cudaError_t err = cudaFree(ptr);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

/**
 * Host main routine
 */
int
main(void)
{
	std::ofstream ofs;
	ofs.open("device_all.txt");
	for(int numElements = 1; numElements < 502267904; numElements*=10) {
		size_t size = numElements * sizeof(float);
		printf("[Vector addition of %d elements]\n", numElements);

		float *h_A = (float *)malloc(size);
		float *h_B = (float *)malloc(size);
		float *h_C = (float *)malloc(size);

		if (h_A == NULL || h_B == NULL || h_C == NULL)
		{
			fprintf(stderr, "Failed to allocate host vectors!\n");
			exit(EXIT_FAILURE);
		}

		for (int i = 0; i < numElements; ++i)
		{
			h_A[i] = rand()/(float)RAND_MAX;
			h_B[i] = rand()/(float)RAND_MAX;
		}

		for(int i = 0; i < 10; i++) {



//			for(int j = 0; j < numElements; j++) {
//				h_C[j] = h_A[j] + h_B[j];
//			}

			timer::start();
			float *d_A = NULL;
			allocateCuda(&d_A, size);

			float *d_B = NULL;
			allocateCuda(&d_B, size);

			float *d_C = NULL;
			allocateCuda(&d_C, size);

			memcopyCudaHostToDevice(h_A, d_A, size);
			memcopyCudaHostToDevice(h_B, d_B, size);


			int threadsPerBlock = 256;
			int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
			vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
			cudaError_t err = cudaGetLastError();

			if (err != cudaSuccess)
			{
				fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

			printf("Copy output data from the CUDA device to the host memory\n");
			memcopyCudaDeviceToHost(h_C, d_C, size);

			freeCuda(d_A);
			freeCuda(d_B);
			freeCuda(d_C);
			timer::stop();

			std::cout << "[Timer]" << timer::read() << std::endl;
			ofs << numElements << "\t" << timer::read() << "\n";
		}

		free(h_A);
		free(h_B);
		free(h_C);
	}
	ofs.close();



    printf("Done\n");
    return 0;
}

