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
		static void run_matrix_addition(T *d_A, T *d_B, T *d_C, std::size_t size) {
			int threadsPerBlock = 32;
			int blocksPerGrid =((size) + threadsPerBlock - 1) / threadsPerBlock;

			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

			cuda_matrix_addition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

			std::cout << "done\n\n";
		}

		template<typename T>
		static void run_matrix_hadamard(T *d_A, T *d_B, T *d_C, std::size_t size) {
			int threadsPerBlock = 32;
			int blocksPerGrid =((size) + threadsPerBlock - 1) / threadsPerBlock;

			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

			cuda_matrix_hadamard<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

			std::cout << "done\n\n";
		}

		template<typename T>
		static void run_vector_dyadic(T *d_A, T *d_B, T *d_C, std::size_t size) {
			std::cout << "Computing result using CUDA Kernel...\n";

			int threadsPerBlock = 32;
			int blocksPerGrid =((size * size) + threadsPerBlock - 1) / threadsPerBlock;
			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
			cuda_vector_dyadic<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

			std::cout << "done\n\n";
		}
	};

	template<>
	struct kernel_dispatcher<grid_definition::TWO_DIM> {

		template<typename T>
		static void run_matrix_addition(T *d_A, T *d_B, T *d_C, std::size_t size) {
			std::cout << "Computing result using CUDA Kernel...\n";

			int threadsPerBlock = 32;
			int blocksPerGrid =((size) + threadsPerBlock - 1) / threadsPerBlock;

			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

			cuda_matrix_addition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

			std::cout << "done\n\n";
		}

		template<typename T>
		static void run_matrix_hadamard(T *d_A, T *d_B, T *d_C, std::size_t size) {
			std::cout << "Computing result using CUDA Kernel...\n";

			int threadsPerBlock = 32;
			int blocksPerGrid =((size) + threadsPerBlock - 1) / threadsPerBlock;

			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

			cuda_matrix_hadamard<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

			std::cout << "done\n\n";
		}

		template<typename T>
		static void run_vector_dyadic(T *d_A, T *d_B, T *d_C, std::size_t size) {
			std::cout << "Computing result using CUDA Kernel...\n";

			int threadsPerBlock = 32;
			int blocksPerGrid =((size * size) + threadsPerBlock - 1) / threadsPerBlock;
			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
			cuda_vector_dyadic<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

			std::cout << "done\n\n";
		}
	};

	template<>
	struct kernel_dispatcher<grid_definition::THREE_DIM> {

		template<typename T>
		static void run_matrix_addition(T *d_A, T *d_B, T *d_C, std::size_t size) {
			std::cout << "Computing result using CUDA Kernel...\n";

			int threadsPerBlock = 32;
			int blocksPerGrid =((size) + threadsPerBlock - 1) / threadsPerBlock;

			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

			cuda_matrix_addition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

			std::cout << "done\n\n";
		}

		template<typename T>
		static void run_matrix_hadamard(T *d_A, T *d_B, T *d_C, std::size_t size) {
			std::cout << "Computing result using CUDA Kernel...\n";

			int threadsPerBlock = 32;
			int blocksPerGrid =((size) + threadsPerBlock - 1) / threadsPerBlock;

			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

			cuda_matrix_hadamard<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

			std::cout << "done\n\n";
		}

		template<typename T>
		static void run_vector_dyadic(T *d_A, T *d_B, T *d_C, std::size_t size) {
			std::cout << "Computing result using CUDA Kernel...\n";

			int threadsPerBlock = 32;
			int blocksPerGrid =((size * size) + threadsPerBlock - 1) / threadsPerBlock;
			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
			cuda_vector_dyadic<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

			std::cout << "done\n\n";
		}
	};

}