#include <stdio.h>
#include <assert.h>

#include "benchmarks/one_dim.h"
#include "benchmarks/two_dim.h"
#include "benchmarks/matrix_multiplication.h"

int main(int argc, char **argv) {
	one_dim_benchmarks_run();
	two_dim_benchmarks_run();
	matrix_multiplication_benchmarks_run();

    return 0;
}
