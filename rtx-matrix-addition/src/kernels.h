#pragma once

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

template< typename T >
__global__ void cuda_matrix_addition(const T *A, const T *B, T *C, unsigned long numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

template< typename T >
__global__ void cuda_matrix_hadamard(const T *A, const T *B, T *C, unsigned long numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] * B[i];
    }
}

template< typename T >
__global__ void cuda_vector_dyadic(const T *A, const T *B, T *C, unsigned long numElements) {
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
