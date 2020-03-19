#pragma once

#include <cstdlib>
#include <iostream>

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

namespace cuda {

    template<typename T, std::size_t N, std::size_t M>
    struct matrix {

        T* data;

        const std::size_t size = N * M;

        matrix() {
           std::cout << "matrix()" << std::endl;
           data = reinterpret_cast<T*>(malloc(size * sizeof(T)));
        }

        matrix(const matrix& rhs) {
        	data = reinterpret_cast<T*>(malloc(size * sizeof(T)));
            for(std::size_t i = 0; i < size; i++) {
               data[i] = rhs.data[i];
            }
        }
        
        matrix(matrix&& rhs) {
        	data = rhs.data;
        	rhs.data = nullptr;
        }

        virtual ~matrix() {
            free(data);
        }

        matrix& operator=(const matrix& rhs) {
        	if(this == rhs)
        		return *this;

            for(std::size_t i = 0; i < size; i++) {
               data[i] = rhs.data[i];
            }

            return *this;
        }

        matrix& operator=(matrix&& rhs) {
        	if(this == rhs)
        		return *this;

        	data = rhs.data;
        	rhs.data = nullptr;

        	return *this;
        }

        bool operator==(const matrix& rhs) const {
        	for(std::size_t i = 0; i < size; i++) {
        		if(data[i] != rhs.data[i])
        			return false;
        	}

        	return true;
        }

        void random_initialize() {
            for(std::size_t i = 0; i < size; i++) {
               data[i] = static_cast<T>(10 * (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)));
            }
        }

        void print() const {
            for(std::size_t i = 0; i < N; i++) {
            	for(std::size_t j = 0; j < M; j++) {
            		std::cout << data[j + i * N] << " ";
            	}
               std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        matrix add_parallel(const matrix& rhs) const;

        matrix add_sequential(const matrix& rhs) const;

        matrix hadamard_parallel(const matrix& rhs) const;

        matrix hadamard_sequential(const matrix& rhs) const;
    };

    template<typename T, std::size_t N, std::size_t M>
    matrix<T, N, M> matrix<T, N, M>::add_parallel(const matrix<T, N, M>& rhs) const {
    	matrix<T, N, M> result;

    	T *d_A, *d_B, *d_C;

		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), size * sizeof(T)));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), size * sizeof(T)));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), size * sizeof(T)));

		checkCudaErrors(cudaMemcpy(d_A, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_B, rhs.data, size * sizeof(T), cudaMemcpyHostToDevice));

		std::cout << "Computing result using CUDA Kernel...\n";

		int threadsPerBlock = 32;
		int blocksPerGrid =((size) + threadsPerBlock - 1) / threadsPerBlock;
		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		cuda_matrix_addition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size * sizeof(T));

		std::cout << "done\n";

		checkCudaErrors(cudaMemcpy(result.data, d_C, size * sizeof(T), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_A));
		checkCudaErrors(cudaFree(d_B));
		checkCudaErrors(cudaFree(d_C));

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
    matrix<T, N, M> matrix<T, N, M>::hadamard_parallel(const matrix<T, N, M>& rhs) const {
    	matrix<T, N, M> result;

    	T *d_A, *d_B, *d_C;

		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), size * sizeof(T)));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), size * sizeof(T)));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), size * sizeof(T)));

		checkCudaErrors(cudaMemcpy(d_A, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_B, rhs.data, size * sizeof(T), cudaMemcpyHostToDevice));

		std::cout << "Computing result using CUDA Kernel...\n";

		int threadsPerBlock = 32;
		int blocksPerGrid =((size) + threadsPerBlock - 1) / threadsPerBlock;
		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		cuda_matrix_hadamard<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size * sizeof(T));

		std::cout << "done\n";

		checkCudaErrors(cudaMemcpy(result.data, d_C, size * sizeof(T), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_A));
		checkCudaErrors(cudaFree(d_B));
		checkCudaErrors(cudaFree(d_C));

		return result;
    }

    template<typename T, std::size_t N, std::size_t M>
    matrix<T, N, M> matrix<T, N, M>::hadamard_sequential(const matrix<T, N, M>& rhs) const {
    	matrix<T, N, M> result;

		for(std::size_t i = 0; i < size; i++) {
			result.data[i] = this->data[i] * rhs.data[i];
		}

		return result;
    }

}
