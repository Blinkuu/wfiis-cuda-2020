#pragma once

#include "matrix.h"

namespace cuda {

	template< typename T >
	__global__ void cuda_matrix_dyadic(const T *A, const T *B, T *C, unsigned long numElements) {
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

	template<typename T, std::size_t Size>
	class vector : public matrix<T, Size, 1> {
	public:
		using matrix<T, Size, 1>::print;

		matrix<T, Size, Size> dyadic_parallel(const vector& rhs) const;

		matrix<T, Size, Size> dyadic_sequential(const vector& rhs) const;
	};

	template<typename T, std::size_t Size>
	matrix<T, Size, Size> vector<T, Size>::dyadic_parallel(const vector<T, Size>& rhs) const {
		matrix<T, Size, Size> result;

    	T *d_A, *d_B, *d_C;

		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), this->size * sizeof(T)));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), this->size * sizeof(T)));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), Size * Size * sizeof(T)));

		checkCudaErrors(cudaMemcpy(d_A, this->data, this->size * sizeof(T), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_B, rhs.data, this->size * sizeof(T), cudaMemcpyHostToDevice));

		std::cout << "Computing result using CUDA Kernel...\n";

		int threadsPerBlock = 32;
		int blocksPerGrid =((this->size) + threadsPerBlock - 1) / threadsPerBlock;
		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		cuda_matrix_dyadic<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Size);

		std::cout << "done\n";

		checkCudaErrors(cudaMemcpy(result.data, d_C, Size * Size * sizeof(T), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_A));
		checkCudaErrors(cudaFree(d_B));
		checkCudaErrors(cudaFree(d_C));

		return result;
	}

	template<typename T, std::size_t Size>
	matrix<T, Size, Size> vector<T, Size>::dyadic_sequential(const vector<T, Size>& rhs) const {
		matrix<T, Size, Size> result;

		for(std::size_t i = 0; i < Size; i++) {
			for(std::size_t j = 0; j < Size; j++) {
				result.data[j + i * Size] = this->data[i] * rhs.data[j];
			}
		}

		return result;
	}

}
