#include <stdio.h>
#include <assert.h>

#include "vector.h"
#include "timer.h"

template<typename T, std::size_t Size>
void matrix_addition() {
	cuda::matrix<T, Size, Size> h_A;
	h_A.random_initialize();

	cuda::matrix<T, Size, Size> h_B;
	h_B.random_initialize();

	cuda::matrix<T, Size, Size> h_C = h_A.template add_parallel <cuda::grid_definition::ONE_DIM>(h_B);
	cuda::matrix<T, Size, Size> d_C = h_A.add_sequential(h_B);

	std::cout << (h_C == d_C ? "Addition passed" : "Addition failed") << "\n\n";
}

template<typename T, std::size_t Size>
void matrix_hadamard() {
	cuda::matrix<T, Size, Size> h_A;
	h_A.random_initialize();

	cuda::matrix<T, Size, Size> h_B;
	h_B.random_initialize();

	cuda::matrix<T, Size, Size> h_C = h_A.template hadamard_parallel<cuda::grid_definition::ONE_DIM>(h_B);
	cuda::matrix<T, Size, Size> d_C = h_A.hadamard_sequential(h_B);

	std::cout << (h_C == d_C ? "Hadamard passed" : "Hadamard failed") << "\n\n";
}

template<typename T, std::size_t Size>
void vector_dyadic() {
	cuda::vector<T, Size> h_A;
	h_A.random_initialize();

	cuda::vector<T, Size> h_B;
	h_B.random_initialize();

	cuda::matrix<T, Size, Size> h_C = h_A.template dyadic_parallel<cuda::grid_definition::ONE_DIM>(h_B);
	cuda::matrix<T, Size, Size> d_C = h_A.dyadic_sequential(h_B);

	std::cout << (h_C == d_C ? "Dyadic passed" : "Dyadic failed") << "\n\n";
}

int main(int argc, char **argv) {
	matrix_addition<float, 100>();
	matrix_addition<int, 100>();

	matrix_hadamard<float, 100>();
	matrix_hadamard<int, 100>();

	vector_dyadic<float, 100>();
	vector_dyadic<int, 100>();

    return 0;
}
