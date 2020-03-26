#pragma once

#include "../maths/vector.h"
#include "../time/timer.h"
#include "../configs/config.h"
#include "get_type_name.h"

#include <fstream>

template<typename T, std::size_t Size>
void matrix_multiplication_test() {
	cuda::matrix<T, Size, Size> h_A;
	h_A.random_initialize();

	cuda::matrix<T, Size, Size> h_B;
	h_B.random_initialize();

	cuda::timer::start();
	cuda::matrix<T, Size, Size> h_C = h_A.template multiply_parallel <cuda::grid_definition::TWO_DIM>(h_B);
	cuda::timer::stop();

	const double parallel_time = cuda::timer::read();
	std::cout << "[PARALLEL MULTIPLICATION 1D] " << parallel_time << " [s]\n";

	cuda::timer::start();
	cuda::matrix<T, Size, Size> d_C = h_A.multiply_sequential(h_B);
	cuda::timer::stop();

	const double sequential_time = cuda::timer::read();
	std::cout << "[SEQUENTIAL MULTIPLICATION 1D] " << sequential_time << " [s]\n";

	std::ofstream ofs;
	ofs.open("matrix_multiplication_" + get_type_name<T>() + ".txt", std::ofstream::out | std::ofstream::app);
	ofs << Size << "\t" << parallel_time << "\t" << sequential_time << std::endl;

	std::cout << (h_C == d_C ? "[MULTIPLICATION PASSED]" : "[MULTIPLICATION FAILED]") << "\n\n";
}

static void matrix_multiplication_benchmarks_run() {
	std::cout << "============== [2D BENCHMARKS] ==============\n\n";

	for(std::size_t i = 0; i < cuda::config::max_iterations; i++) {
		matrix_multiplication_test<double, cuda::config::num_elements>();
		matrix_multiplication_test<int, cuda::config::num_elements>();
	}
}

