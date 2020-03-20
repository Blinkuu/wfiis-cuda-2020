#pragma once

#include "grid.h"
#include "kernels.h"

namespace cuda {

	template<grid_definition Definition>
	struct kernel_dispatcher {
		static void run_matrix_addition() = delete;
		static void run_matrix_hadamard() = delete;
		static void run_vector_dyadic() = delete;
	};

	template<>
	struct kernel_dispatcher<grid_definition::ONE_DIM> {
		template<typename T>
		static void run_matrix_addition(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int threadsPerBlock = 1024;
			int blocksPerGrid =((N * M) + threadsPerBlock - 1) / threadsPerBlock;

			cuda_matrix_addition_1d<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N * M);
		}

		template<typename T>
		static void run_matrix_hadamard(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int threadsPerBlock = 1024;
			int blocksPerGrid =((N * M) + threadsPerBlock - 1) / threadsPerBlock;

			cuda_matrix_hadamard_1d<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N *M);
		}

		template<typename T>
		static void run_vector_dyadic(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int threadsPerBlock = 1024;
			int blocksPerGrid =((N * M) + threadsPerBlock - 1) / threadsPerBlock;

			cuda_vector_dyadic_1d<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
		}
	};

	template<>
	struct kernel_dispatcher<grid_definition::TWO_DIM> {

		template<typename T>
		static void run_matrix_addition(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			const int threads_per_block = 32;
			const dim3 block_size(threads_per_block, threads_per_block);
			const int block_x = (N + threads_per_block - 1)/threads_per_block;
			const int block_y = (M + threads_per_block - 1)/threads_per_block;
			const dim3 grid_size = dim3(block_x, block_y);

			cuda_matrix_addition_2d<<<grid_size, block_size>>>(d_A, d_B, d_C, N, M);
		}

		template<typename T>
		static void run_matrix_hadamard(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			const int threads_per_block = 32;
			const dim3 block_size(threads_per_block, threads_per_block);
			const int block_x = (N + threads_per_block - 1)/threads_per_block;
			const int block_y = (M + threads_per_block - 1)/threads_per_block;
			const dim3 grid_size = dim3(block_x, block_y);

			cuda_matrix_hadamard_2d<<<grid_size, block_size>>>(d_A, d_B, d_C, N, M);
		}

		template<typename T>
		static void run_vector_dyadic(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int threadsPerBlock = 1024;
			int blocksPerGrid =((N * M) + threadsPerBlock - 1) / threadsPerBlock;

			cuda_vector_dyadic_2d<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, 1);
		}
	};

	template<>
	struct kernel_dispatcher<grid_definition::THREE_DIM> {

		template<typename T>
		static void run_matrix_addition(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int threadsPerBlock = 1024;
			int blocksPerGrid =((N * M) + threadsPerBlock - 1) / threadsPerBlock;

			cuda_matrix_addition_3d<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M);
		}

		template<typename T>
		static void run_matrix_hadamard(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int threadsPerBlock = 1024;
			int blocksPerGrid =((N * M) + threadsPerBlock - 1) / threadsPerBlock;

			cuda_matrix_hadamard_3d<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M);
		}

		template<typename T>
		static void run_vector_dyadic(T *d_A, T *d_B, T *d_C, std::size_t N, std::size_t M) {
			int threadsPerBlock = 1024;
			int blocksPerGrid =(N * M + threadsPerBlock - 1) / threadsPerBlock;

			cuda_vector_dyadic_3d<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M);
		}
	};

}
