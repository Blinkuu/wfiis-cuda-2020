#include <stdio.h>
#include <assert.h>

#include "matrix.h"

template<typename T, std::size_t Size>
void matrix_addition() {
	cuda::matrix<T, Size, Size> h_A;
	h_A.random_initialize();

	cuda::matrix<T, Size, Size> h_B;
	h_B.random_initialize();

	cuda::matrix<T, Size, Size> h_C = h_A.add_parallel(h_B);
	cuda::matrix<T, Size, Size> d_C = h_A.add_sequential(h_B);

	std::cout << (h_C == d_C ? "Addition passed" : "Addition failed") << std::endl;
}

template<typename T, std::size_t Size>
void matrix_hadamard() {
	cuda::matrix<T, Size, Size> h_A;
	h_A.random_initialize();

	cuda::matrix<T, Size, Size> h_B;
	h_B.random_initialize();

	cuda::matrix<T, Size, Size> h_C = h_A.hadamard_parallel(h_B);
	cuda::matrix<T, Size, Size> d_C = h_A.hadamard_sequential(h_B);

	std::cout << (h_C == d_C ? "Hadamard passed" : "Hadamard failed") << std::endl;
}

int main(int argc, char **argv) {
	matrix_addition<float, 100>();
	matrix_addition<int, 100>();

	matrix_hadamard<float, 100>();
	matrix_hadamard<int, 100>();

    return 0;
}

