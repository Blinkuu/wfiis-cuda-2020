#pragma once

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

// 1D

template< typename T >
__global__ void cuda_matrix_addition_1d(const T *A, const T *B, T *C, unsigned long numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

template< typename T >
__global__ void cuda_matrix_hadamard_1d(const T *A, const T *B, T *C, unsigned long numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] * B[i];
    }
}

template< typename T >
__global__ void cuda_vector_dyadic_1d(const T *A, const T *B, T *C, unsigned long numElements) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements * numElements)
	{
		for(int k = 0; k < numElements; k++) {
			for(int m = 0; m < numElements; m++) {
				if((m + k * numElements) == i) {
					C[i] = A[k] * B[m];
				}
			}
		}

	}
}

// 2D

template< typename T >
__global__ void cuda_matrix_addition_2d(const T *A, const T *B, T *C, unsigned long N, unsigned long M) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	int k = i + j * M;

	if(i < M && j < N) {
		C[k] = A[k] + B[k];
	}
}

template< typename T >
__global__ void cuda_matrix_hadamard_2d(const T *A, const T *B, T *C, unsigned long N, unsigned long M) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	int k = i + j * M;

	if(i < M && j < N) {
		C[k] = A[k] * B[k];
	}
}

template< typename T >
__global__ void cuda_vector_dyadic_2d(const T *A, const T *B, T *C, unsigned long N, unsigned long M) {
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	int k = i + j * M;

	if(i < M && j < N) {
		C[k] = A[i] * B[j];
	}
}

// 3D

template< typename T >
__global__ void cuda_matrix_addition_3d(const T *A, const T *B, T *C, unsigned long numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

template< typename T >
__global__ void cuda_matrix_hadamard_3d(const T *A, const T *B, T *C, unsigned long numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] * B[i];
    }
}

template< typename T >
__global__ void cuda_vector_dyadic_3d(const T *A, const T *B, T *C, unsigned long numElements) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements * numElements)
	{
		for(int k = 0; k < numElements; k++) {
			for(int m = 0; m < numElements; m++) {
				if((m + k * numElements) == i) {
					C[i] = A[k] * B[m];
				}
			}
		}

	}
}
