#pragma once

#include "grid.h"
#include "kernels.h"
#include "../configs/config.h"

namespace cuda {

	template<grid_definition Definition>
	struct kernel_dispatcher {
		static void run_matrix_addition() = delete;
		static void run_matrix_hadamard() = delete;
		static void run_matrix_multiply() = delete;
		static void run_vector_dyadic() = delete;
	};

	template<>
	struct kernel_dispatcher<grid_definition::ONE_DIM> {
		template<typename T>
		static void run_matrix_addition(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int blocksPerGrid =((N * M) + config::threads_per_block - 1) / config::threads_per_block;

			cuda_matrix_addition_1d<<<blocksPerGrid, config::threads_per_block>>>(d_A, d_B, d_C, N * M);
		}

		template<typename T>
		static void run_matrix_hadamard(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int blocksPerGrid =((N * M) + config::threads_per_block - 1) / config::threads_per_block;

			cuda_matrix_hadamard_1d<<<blocksPerGrid, config::threads_per_block>>>(d_A, d_B, d_C, N *M);
		}

		template<typename T>
		static void run_vector_dyadic(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int blocksPerGrid =((N * M) + config::threads_per_block - 1) / config::threads_per_block;

			cuda_vector_dyadic_1d<<<blocksPerGrid, config::threads_per_block>>>(d_A, d_B, d_C, N);
		}
	};

	template<>
	struct kernel_dispatcher<grid_definition::TWO_DIM> {

		template<typename T>
		static void run_matrix_addition(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			const dim3 block_size(config::threads_per_block, config::threads_per_block);
			const int block_x = (N + config::threads_per_block - 1)/config::threads_per_block;
			const int block_y = (M + config::threads_per_block - 1)/config::threads_per_block;
			const dim3 grid_size = dim3(block_x, block_y);

			cuda_matrix_addition_2d<<<grid_size, block_size>>>(d_A, d_B, d_C, N, M);
		}

		template<typename T>
		static void run_matrix_hadamard(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			const dim3 block_size(config::threads_per_block, config::threads_per_block);
			const int block_x = (N + config::threads_per_block - 1)/config::threads_per_block;
			const int block_y = (M + config::threads_per_block - 1)/config::threads_per_block;
			const dim3 grid_size = dim3(block_x, block_y);

			cuda_matrix_hadamard_2d<<<grid_size, block_size>>>(d_A, d_B, d_C, N, M);
		}

		template<typename T>
		static void run_matrix_multiply(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			const dim3 block_size(config::threads_per_block, config::threads_per_block);
			const int block_x = (N + config::threads_per_block - 1)/config::threads_per_block;
			const int block_y = (M + config::threads_per_block - 1)/config::threads_per_block;
			const dim3 grid_size = dim3(block_x, block_y);

			cuda_matrix_multiplication_2d<<<grid_size, block_size>>>(d_A, d_B, d_C, N, M);
		}

		template<typename T>
		static void run_vector_dyadic(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			const dim3 block_size(config::threads_per_block, config::threads_per_block);
			const int block_x = (N + config::threads_per_block - 1)/config::threads_per_block;
			const int block_y = (N + config::threads_per_block - 1)/config::threads_per_block;
			const dim3 grid_size = dim3(block_x, block_y);

			cuda_vector_dyadic_2d<<<grid_size, block_size>>>(d_A, d_B, d_C, N, N);
		}
	};

}
