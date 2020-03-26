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

		kernel_dispatcher<Definition>::run_vector_dyadic(data, rhs.data, result.data, Size, Size);
		checkCudaErrors(cudaGetLastError());

		return result;
	}

	template<typename T, std::size_t Size>
	matrix<T, Size, Size> vector<T, Size>::dyadic_sequential(const vector<T, Size>& rhs) const {
		matrix<T, Size, Size> result;

		for(std::size_t i = 0; i < Size; i++) {
			for(std::size_t j = 0; j < Size; j++) {
				result.data[i + j * Size] = this->data[i] * rhs.data[j];
			}
		}

		return result;
	}

}
