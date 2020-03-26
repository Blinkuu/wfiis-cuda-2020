#pragma once

#include <cstdlib>
#include <cmath>
#include <iostream>

#include "../cuda/kernel_dispatcher.h"

namespace cuda {

    template<typename T, std::size_t N, std::size_t M>
    class matrix {
    public:
        T* data;

        const std::size_t size = N * M;

        matrix();

        matrix(const matrix& rhs);

        matrix(matrix&& rhs);

        virtual ~matrix();

        matrix& operator=(const matrix& rhs);
        
        matrix& operator=(matrix&& rhs) noexcept;

        bool operator==(const matrix& rhs) const;

        void random_initialize();

        void print() const;

        template<grid_definition Definition>
        matrix add_parallel(const matrix& rhs) const;

        matrix add_sequential(const matrix& rhs) const;

        template<grid_definition Definition>
        matrix hadamard_parallel(const matrix& rhs) const;

        matrix hadamard_sequential(const matrix& rhs) const;

        template<grid_definition Definition>
        matrix<T, N, N> multiply_parallel(const matrix<T, M, N>& rhs) const;

        matrix<T, N, N> multiply_sequential(const matrix<T, M, N>& rhs) const;
    };

    template<typename T, std::size_t N, std::size_t M>
    matrix<T, N, M>::matrix() {
    	checkCudaErrors(cudaMallocManaged(&data, size * sizeof(T)));
	}

    template<typename T, std::size_t N, std::size_t M>
    matrix<T, N, M>::matrix(const matrix<T, N, M>& rhs) {
		data = reinterpret_cast<T*>(malloc(size * sizeof(T)));
		for(std::size_t i = 0; i < size; i++) {
		   data[i] = rhs.data[i];
		}
	}

	template<typename T, std::size_t N, std::size_t M>
	matrix<T, N, M>::matrix(matrix<T, N, M>&& rhs) {
		data = rhs.data;
		rhs.data = nullptr;
	}

	template<typename T, std::size_t N, std::size_t M>
	matrix<T, N, M>::~matrix() {
		checkCudaErrors(cudaFree(data));
	}

	template<typename T, std::size_t N, std::size_t M>
	matrix<T, N, M>& matrix<T, N, M>::operator=(const matrix<T, N, M>& rhs) {
		if(this == rhs)
			return *this;

		for(std::size_t i = 0; i < size; i++) {
		   data[i] = rhs.data[i];
		}

		return *this;
	}

	template<typename T, std::size_t N, std::size_t M>
	matrix<T, N, M>& matrix<T, N, M>::operator=(matrix<T, N, M>&& rhs) noexcept {
		if(this == rhs)
			return *this;

		data = rhs.data;
		rhs.data = nullptr;

		return *this;
	}

	template<typename T, std::size_t N, std::size_t M>
	bool matrix<T, N, M>::operator==(const matrix<T, N, M>& rhs) const {
		for(std::size_t i = 0; i < size; i++) {
			if(data[i] - rhs.data[i] > 1e-4)
				return false;
		}

		return true;
	}

	template<typename T, std::size_t N, std::size_t M>
	void matrix<T, N, M>::random_initialize() {
		for(std::size_t i = 0; i < size; i++) {
		   data[i] = static_cast<T>(10 * (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)));
		}
	}

	template<typename T, std::size_t N, std::size_t M>
	void matrix<T, N, M>::print() const {
		for(std::size_t i = 0; i < N; i++) {
			for(std::size_t j = 0; j < M; j++) {
				std::cout << data[i + j * N] << " ";
			}
		   std::cout << std::endl;
		}
		std::cout << std::endl;
	}

    template<typename T, std::size_t N, std::size_t M>
    template<grid_definition Definition>
    matrix<T, N, M> matrix<T, N, M>::add_parallel(const matrix<T, N, M>& rhs) const {
    	matrix<T, N, M> result;

		kernel_dispatcher<Definition>::run_matrix_addition(data, rhs.data, result.data, N, M);
		checkCudaErrors(cudaGetLastError());

		return result;
    }

    template<typename T, std::size_t N, std::size_t M>
    matrix<T, N, M> matrix<T, N, M>::add_sequential(const matrix<T, N, M>& rhs) const {
    	matrix<T, N, M> result;

		for(std::size_t i = 0; i < size; i++) {
			result.data[i] = this->data[i] + rhs.data[i];
		}

		return result;
    }

    template<typename T, std::size_t N, std::size_t M>
    template<grid_definition Definition>
    matrix<T, N, M> matrix<T, N, M>::hadamard_parallel(const matrix<T, N, M>& rhs) const {
    	matrix<T, N, M> result;

		kernel_dispatcher<Definition>::run_matrix_hadamard(data, rhs.data, result.data, N, M);
		checkCudaErrors(cudaGetLastError());

		return result;
    }

    template<typename T, std::size_t N, std::size_t M>
    matrix<T, N, M> matrix<T, N, M>::hadamard_sequential(const matrix<T, N, M>& rhs) const {
    	matrix<T, N, M> result;

		for(std::size_t i = 0; i < size; i++) {
			result.data[i] = data[i] * rhs.data[i];
		}

		return result;
    }

    template<typename T, std::size_t N, std::size_t M>
    template<grid_definition Definition>
    matrix<T, N, N> matrix<T, N, M>::multiply_parallel(const matrix<T, M, N>& rhs) const {
    	matrix<T, N, M> result;

    	kernel_dispatcher<Definition>::run_matrix_multiply(data, rhs.data, result.data, N, M);
    	checkCudaErrors(cudaGetLastError());

		return result;
    }

    template<typename T, std::size_t N, std::size_t M>
    matrix<T, N, N> matrix<T, N, M>::multiply_sequential(const matrix<T, M, N>& rhs) const {
    	matrix<T, N, N> result;

		for(std::size_t i = 0; i < N; i++) {
			for(std::size_t j = 0; j < M; j++) {
				T sum{};
				for(int k = 0; k < M; k++){
					sum += data[i + k * M] * rhs.data[k + j * M];
				}
				result.data[i + j * M] = sum;
			}
		}

		return result;
    }

}
