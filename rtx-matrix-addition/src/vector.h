#pragma once

#include "matrix.h"

namespace cuda {

	template<typename T, std::size_t Size>
	struct vector : public matrix<T, Size, 1> {
		using matrix<T, Size, 1>::print;

		matrix<T, Size, Size> dyadic_parallel(const vector& rhs) const;

		matrix<T, Size, Size> dyadic_sequential(const vector& rhs) const {
			matrix<T, Size, Size> result;

			print();
			rhs.print();

			for(std::size_t i = 0; i < Size; i++) {
				for(std::size_t j = 0; j < Size; j++) {
					result.data[j + i * Size] = this->data[i] * rhs.data[j];
				}
			}

			result.print();

			return result;
		}
	};

	template<typename T, std::size_t Size>
	matrix<T, Size, Size> vector<T, Size>::dyadic_parallel(const vector<T, Size>& rhs) const {
		matrix<T, Size, Size> result;

		return result;
	}

//	template<typename T, std::size_t Size>
//	matrix<T, Size, Size> vector<T, Size>::dyadic_sequential(const vector<T, Size>& rhs) {
//
//	}

}
