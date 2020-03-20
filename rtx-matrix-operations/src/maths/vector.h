#pragma once

#include "matrix.h"

namespace cuda {

	template<typename T, std::size_t Size>
	class vector : public matrix<T, Size, 1> {
	public:
		using matrix<T, Size, 1>::print;

		template<grid_definition Definition>
		matrix<T, Size, Size> dyadic_parallel(const vector& rhs) const;

		matrix<T, Size, Size> dyadic_sequential(const vector& rhs) const;
	};

	template<typename T, std::size_t Size>
	template<grid_definition Definition>
	matrix<T, Size, Size> vector<T, Size>::dyadic_parallel(const vector<T, Size>& rhs) const {
		matrix<T, Size, Size> result;

    	T *d_A, *d_B, *d_C;

		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), this->size * sizeof(T)));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), this->size * sizeof(T)));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), Size * Size * sizeof(T)));

		checkCudaErrors(cudaMemcpy(d_A, this->data, this->size * sizeof(T), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_B, rhs.data, this->size * sizeof(T), cudaMemcpyHostToDevice));

		kernel_dispatcher<Definition>::run_vector_dyadic(d_A, d_B, d_C, Size, Size);

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
