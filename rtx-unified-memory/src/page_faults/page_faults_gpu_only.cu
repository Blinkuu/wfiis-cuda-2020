/// managed mamory analysis - cuda lab gpu only mamory access

#include "page_faults.h"
#include <iostream>

__global__
static void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

static void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

void test_page_fault_gpu_only()
{
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);
  cuda::timer::start();
  deviceKernel<<<256, 256>>>(a, N);
  cudaDeviceSynchronize();

  cudaFree(a);
  cuda::timer::stop();
  std::cout << "[DONE] test_page_fault_gpu_only\n";
  std::cout << "[TIME] " << cuda::timer::read() << " s\n";
}

