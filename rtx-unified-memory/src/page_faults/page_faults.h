#pragma once

#include "../time/time.h"
#include <iostream>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

constexpr static int N = 2<<29;

void test_page_fault_gpu_only();
void test_page_fault_cpu_only();
void test_page_fault_cpu_gpu();
void test_page_fault_gpu_cpu();

inline void check(int* a, int N) {
	for(int i = 0; i < N; i++) {
		if(a[i] != 1){
			std::cout << "[FAILED]\n";
			return;
		}
	}
	std::cout << "[PASSED]\n";
}
