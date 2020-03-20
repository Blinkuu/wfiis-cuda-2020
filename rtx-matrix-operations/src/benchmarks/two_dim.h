#pragma once

#include "../maths/vector.h"
#include "../time/timer.h"
#include "get_type_name.h"
#include "config.h"

#include <fstream>

template<typename T, std::size_t Size>
void two_dim_matrix_addition() {
	cuda::matrix<T, Size, Size> h_A;
	h_A.random_initialize();

	cuda::matrix<T, Size, Size> h_B;
	h_B.random_initialize();

	cuda::timer::start();
	cuda::matrix<T, Size, Size> h_C = h_A.template add_parallel <cuda::grid_definition::TWO_DIM>(h_B);
	cuda::timer::stop();

	const double parallel_time = cuda::timer::read();
	std::cout << "[PARALLEL ADD 2D] " << parallel_time << " [s]\n";

	cuda::timer::start();
	cuda::matrix<T, Size, Size> d_C = h_A.add_sequential(h_B);
	cuda::timer::stop();

	const double sequential_time = cuda::timer::read();
	std::cout << "[SEQUENTIAL ADD 2D] " << sequential_time << " [s]\n";

	std::ofstream ofs;
	ofs.open("two_dim_matrix_addition_" + get_type_name<T>() + ".txt", std::ofstream::out | std::ofstream::app);
	ofs << Size << "\t" << parallel_time << "\t" << sequential_time << std::endl;

	std::cout << (h_C == d_C ? "[ADDITION PASSED]" : "[ADDITION FAILED]") << "\n\n";
}

template<typename T, std::size_t Size>
void two_dim_matrix_hadamard() {
	cuda::matrix<T, Size, Size> h_A;
	h_A.random_initialize();

	cuda::matrix<T, Size, Size> h_B;
	h_B.random_initialize();

	cuda::timer::start();
	cuda::matrix<T, Size, Size> h_C = h_A.template hadamard_parallel<cuda::grid_definition::TWO_DIM>(h_B);
	cuda::timer::stop();

	const double parallel_time = cuda::timer::read();
	std::cout << "[PARALLEL HADAMARD 2D] " << parallel_time << " [s]\n";

	cuda::timer::start();
	cuda::matrix<T, Size, Size> d_C = h_A.hadamard_sequential(h_B);
	cuda::timer::stop();

	const double sequential_time = cuda::timer::read();
	std::cout << "[SEQUENTIAL HADAMARD 2D] " << sequential_time << " [s]\n";

	std::ofstream ofs;
	ofs.open("two_dim_matrix_hadamard_" + get_type_name<T>() + ".txt", std::ofstream::out | std::ofstream::app);
	ofs << Size << "\t" << parallel_time << "\t" << sequential_time << std::endl;

	std::cout << (h_C == d_C ? "[HADAMARD PASSED]" : "[HADAMARD FAILED]") << "\n\n";
}

template<typename T, std::size_t Size>
void two_dim_vector_dyadic() {
	cuda::vector<T, Size> h_A;
	h_A.random_initialize();

	cuda::vector<T, Size> h_B;
	h_B.random_initialize();

	cuda::timer::start();
	cuda::matrix<T, Size, Size> h_C = h_A.template dyadic_parallel<cuda::grid_definition::TWO_DIM>(h_B);
	cuda::timer::stop();

	const double parallel_time = cuda::timer::read();
	std::cout << "[PARALLEL DYADIC 2D] " << parallel_time << " [s]\n";

	cuda::timer::start();
	cuda::matrix<T, Size, Size> d_C = h_A.dyadic_sequential(h_B);
	cuda::timer::stop();

	const double sequential_time = cuda::timer::read();
	std::cout << "[SEQUENTIAL DYADIC 2D] " << sequential_time << " [s]\n";

	std::ofstream ofs;
	ofs.open("two_dim_vector_dyadic_" + get_type_name<T>() + ".txt", std::ofstream::out | std::ofstream::app);
	ofs << Size << "\t" << parallel_time << "\t" << sequential_time << std::endl;

	std::cout << (h_C == d_C ? "[DYADIC PASSED]" : "[DYADIC FAILED]") << "\n\n";
}

static void two_dim_benchmarks_run() {
	std::cout << "============== [2D BENCHMARKS] ==============\n\n";

	for(std::size_t i = 0; i < cuda::config::max_iterations; i++) {
		two_dim_matrix_addition<float, cuda::config::num_elements>();
		two_dim_matrix_addition<int, cuda::config::num_elements>();
	}

	for(std::size_t i = 0; i < cuda::config::max_iterations; i++) {
		two_dim_matrix_hadamard<float, cuda::config::num_elements>();
		two_dim_matrix_hadamard<int, cuda::config::num_elements>();
	}

	for(std::size_t i = 0; i < cuda::config::max_iterations; i++) {
		two_dim_vector_dyadic<float, cuda::config::num_elements>();
		two_dim_vector_dyadic<int, cuda::config::num_elements>();
	}
}
