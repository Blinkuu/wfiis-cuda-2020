#include <stdio.h>
#include <math.h>

#include "timer.h"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#define float double

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

constexpr int nstep = 200; // number of time steps

// Specify our 2D dimensions
constexpr int ni = 200;
constexpr int nj = 100;
constexpr float tfac = 8.418e-5; // thermal diffusivity of silver

__global__
void calculate_temp(const float* temp_in, float* temp_out) {
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int j = blockIdx.y*blockDim.y + threadIdx.y;

	if(i < ni-1 && j < nj-1 && i > 0 && j > 0) {
		const int i00 = I2D(ni, i, j);
		const int im10 = I2D(ni, i-1, j);
		const int ip10 = I2D(ni, i+1, j);
		const int i0m1 = I2D(ni, i, j-1);
		const int i0p1 = I2D(ni, i, j+1);

		// evaluate derivatives
		const float d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
		const float d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

		// update temperatures
		temp_out[i00] = temp_in[i00]+tfac*(d2tdx2 + d2tdy2);
	}
}

void step_kernel_mod(float* temp_in, float* temp_out)
{
	constexpr int threads_per_block = 32;

	const dim3 block_size(threads_per_block, threads_per_block);
	const int block_x = (ni + threads_per_block - 1)/threads_per_block;
	const int block_y = (nj + threads_per_block - 1)/threads_per_block;
	const dim3 grid_size = dim3(block_x, block_y);

	calculate_temp<<<grid_size, block_size>>>(temp_in, temp_out);
	cudaDeviceSynchronize();
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  // loop over all points in domain (except boundary)
  for ( int j=1; j < nj-1; j++ ) {
    for ( int i=1; i < ni-1; i++ ) {
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
  }
}

int main()
{
  float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;

  const int size = ni * nj * sizeof(float);

  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);
  checkCudaErrors(cudaMallocManaged(&temp1, size));
  checkCudaErrors(cudaMallocManaged(&temp2, size));

  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }

  cuda::timer::start();
  for (int istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp;
  }

  cuda::timer::stop();
  printf("cpu-time: %f\n", cuda::timer::read());

  cuda::timer::start();
  for (int istep=0; istep < nstep; istep++) {
    step_kernel_mod(temp1, temp2);

    temp_tmp = temp1;
    temp1 = temp2;
    temp2= temp_tmp;
  }
  cuda::timer::stop();
  printf("gpu-time: %f\n", cuda::timer::read());

  float maxError = 0.0f;
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1[i]-temp1_ref[i]); }
  }

  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  free( temp1_ref );
  free( temp2_ref );
  free( temp1 );
  free( temp2 );

  return 0;
}
