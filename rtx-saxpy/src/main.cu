#include <stdio.h>
#include <cstdlib>
#include "timer.h"

#define N (1 << 20)// 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively
 * and use profiler to check your progress
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 25us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(int * __restrict__ a, int * __restrict__ b, int * __restrict__ result)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

//	if(index < N)
//		result[index] = 2 * a[index] + b[index];

	const unsigned int s = blockDim.x * gridDim.x;
		while( i + s * 2 < N )
		{
			result[i] = 2 * a[i] + b[i];
			i += s;
			result[i] = 2 * a[i] + b[i];
			i += s;
			result[i] = 2 * a[i] + b[i];
			i += s;
		}
		while(	i < N  	)
		{
			result[i] = 2 * a[i] + b[i];
			i += s;
		}

}


__global__ void init(int * a, int val)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if ( tid < N )
        a[tid] = val;
}


int main()
{
    int *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector

    int deviceId;
	cudaGetDevice(&deviceId);

    cuda::timer::start();

    a = (int*) malloc(size);
    b = (int*) malloc(size);
    c = (int*) malloc(size);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

	cudaMemPrefetchAsync(a, size, deviceId);
	cudaMemPrefetchAsync(b, size, deviceId);
	cudaMemPrefetchAsync(c, size, deviceId);

    int threads_per_block = 64;
    int number_of_blocks = 256; //(N + threads_per_block - 1) / threads_per_block;//(N / threads_per_block) + 1;

    init <<< number_of_blocks, threads_per_block >>> ( a, 2 );
	init <<< number_of_blocks, threads_per_block >>> ( b, 1 );
	init <<< number_of_blocks, threads_per_block >>> ( c, 0 );

	saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );


    cudaDeviceSynchronize();
//    for(int i = 0; i < N; i++){
//    	a[i] = 2;
//    	b[i] = 1;
//    	c[i] = a[i] * 2 + b[i];
//    }

    cuda::timer::stop();
    printf("time: %f s\n", cuda::timer::read());

    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    //cudaFree( a ); cudaFree( b ); cudaFree( c );

}


//#include <stdio.h>
//
//__global__
//void saxpy(int n, float a, float *x, float *y)
//{
//  int i = blockIdx.x*blockDim.x + threadIdx.x;
//  if (i < n) y[i] = a*x[i] + y[i];
//}
//
//int main(void)
//{
//  int N = 1<<20;
//  float *x, *y, *d_x, *d_y;
//  x = (float*)malloc(N*sizeof(float));
//  y = (float*)malloc(N*sizeof(float));
//
//  for (int i = 0; i < N; i++) {
//    x[i] = 1.0f;
//    y[i] = 2.0f;
//  }
//
//  cuda::timer::start();
//  cudaMalloc(&d_x, N*sizeof(float));
//  cudaMalloc(&d_y, N*sizeof(float));
//
//  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
//  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
//
//  // Perform SAXPY on 1M elements
//  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
//
//  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
//
//  cuda::timer::stop();
//
//  printf("time: %f\n", cuda::timer::read());
//
//  float maxError = 0.0f;
//  for (int i = 0; i < N; i++)
//    maxError = max(maxError, abs(y[i]-4.0f));
//  printf("Max error: %f\n", maxError);
//
//  cudaFree(d_x);
//  cudaFree(d_y);
//  free(x);
//  free(y);
//}
