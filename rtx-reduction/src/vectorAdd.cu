#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "timer.h"

#define WHOLE 1

const int THREADS_PER_BLOCK = 256;

__global__ void warmStart()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

__global__ void reduce_naive(int* input, int size)
{
    __shared__ int partialSum[2 * THREADS_PER_BLOCK];

    partialSum[threadIdx.x] = 0;
    __syncthreads();

    std::size_t t = threadIdx.x;
    std::size_t start = 2 * blockIdx.x * blockDim.x;

    if (t < size && start < size) {
            partialSum[t] = input[start + t];
            partialSum[blockDim.x + t] = input[start + blockDim.x + t];
    
            for (std::size_t stride = 1; stride <= blockDim.x; stride *= 2) {
                        __syncthreads();
                        if (t % stride == 0) {
                                        partialSum[2 * t] += partialSum[2 * t + stride];
                                    }
                    }
        }

    if (t == 0) {
            input[blockIdx.x] = partialSum[0];
        }
}

__global__ void reduce_optimized(int* input, int* output, std::size_t size)
{
    __shared__ int partialSum[THREADS_PER_BLOCK];

    partialSum[threadIdx.x] = 0;
    __syncthreads();

    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_id < size) {
            partialSum[threadIdx.x] = input[t_id];
            __syncthreads();
    
            for (int s = blockDim.x / 2; s > 0; s /= 2) {
                        int index = threadIdx.x;
                        if (index < s) {
                                        partialSum[index] += partialSum[index + s];
                                    }
            
                        __syncthreads();
                    }
    
            if (threadIdx.x == 0) {
                        output[blockIdx.x] = partialSum[0];
                    }
        }
}

int test_reduce_naive(int* input, std::size_t size, int deviceId)
{
#if WHOLE
            cuda::timer::start();
    
            cudaMemPrefetchAsync(input, size * sizeof(int), deviceId);
            cudaDeviceSynchronize();
    
            int BLOCKS_PER_GRID = std::ceil(size / static_cast<double>(THREADS_PER_BLOCK));
            reduce_naive<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(input, size);
            cudaDeviceSynchronize();
    
            int sum = 0;
            for (int i = 0; i < BLOCKS_PER_GRID; i++) {
                        sum += input[i];
                    }
    
            cuda::timer::stop();
            printf("[NAIVE] time: %f s\n", cuda::timer::read());
            printf("[NAIVE] sum: %d\n\n", sum);

            return sum;
#else
    
            cudaMemPrefetchAsync(input, size * sizeof(int), deviceId);
            cudaDeviceSynchronize();
            cuda::timer::start();
    
            int BLOCKS_PER_GRID = std::ceil(size / static_cast<double>(THREADS_PER_BLOCK));
            reduce_naive<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(input, size);
            cudaDeviceSynchronize();
            cuda::timer::stop();
    
            int sum = 0;
            for (int i = 0; i < BLOCKS_PER_GRID; i++) {
                        sum += input[i];
                    }
    
            printf("[NAIVE KERNEL] time: %f s\n", cuda::timer::read());
            printf("[NAIVE KERNEL] sum: %d\n\n", sum);
            
            return sum;
#endif
}

int test_reduce_optimized(int* input, std::size_t size, int deviceId)
{
#if WHOLE
            cuda::timer::start();
    
            cudaMemPrefetchAsync(input, size * sizeof(int), deviceId);
            cudaDeviceSynchronize();
    
            int BLOCKS_PER_GRID = std::ceil(size / static_cast<double>(THREADS_PER_BLOCK));
            reduce_optimized<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(input, input,
                                size);
            cudaDeviceSynchronize();
    
            size = BLOCKS_PER_GRID;
            BLOCKS_PER_GRID = std::ceil(BLOCKS_PER_GRID / static_cast<double>(THREADS_PER_BLOCK));
            while (BLOCKS_PER_GRID > 1) {
                        reduce_optimized<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(input, input,
                                                size);
                        cudaDeviceSynchronize();
            
                        size = BLOCKS_PER_GRID;
                        BLOCKS_PER_GRID = std::ceil(BLOCKS_PER_GRID / static_cast<double>(THREADS_PER_BLOCK));
                    }
    
            reduce_optimized<<<1, THREADS_PER_BLOCK>>>(input, input, size);
            int sum = input[0];
    
            cuda::timer::stop();
    
            printf("[OPTIMIZED] time: %f s\n", cuda::timer::read());
            printf("[OPTIMIZED] sum: %d\n\n", sum);

            return sum;
#else
            cudaMemPrefetchAsync(input, size * sizeof(int), deviceId);
            cudaDeviceSynchronize();
    
            cuda::timer::start();
            int BLOCKS_PER_GRID = std::ceil(size / static_cast<double>(THREADS_PER_BLOCK));
            reduce_optimized<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(input, input,
                                size);
            cudaDeviceSynchronize();
    
            size = BLOCKS_PER_GRID;
            BLOCKS_PER_GRID = std::ceil(BLOCKS_PER_GRID / static_cast<double>(THREADS_PER_BLOCK));
            while (BLOCKS_PER_GRID > 1) {
                        reduce_optimized<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(input, input,
                                                size);
                        cudaDeviceSynchronize();
            
                        size = BLOCKS_PER_GRID;
                        BLOCKS_PER_GRID = std::ceil(BLOCKS_PER_GRID / static_cast<double>(THREADS_PER_BLOCK));
                    }
    
            reduce_optimized<<<1, THREADS_PER_BLOCK>>>(input, input, size);
            cuda::timer::stop();
    
            int sum = input[0];
    
            printf("[OPTIMIZED KERNEL] time: %f s\n", cuda::timer::read());
            printf("[OPTIMIZED KERNEL] sum: %d\n\n", sum);

            return sum;
#endif
}

int test_cpu(int* input, std::size_t size)
{
    cuda::timer::start();

    int sum = std::accumulate(input, input + size, 0);
    ;

    cuda::timer::stop();
    printf("[CPU] time: %f s\n", cuda::timer::read());
    printf("[CPU] sum: %d\n\n", sum);

    return sum;
}

int main(void)
{
    int deviceId;
    cudaGetDevice(&deviceId);

    warmStart<<<1, 1>>>(); 
    
    const std::size_t size = 1024 * 65535;

    int* input_gpu_naive = nullptr;
    int* input_gpu_optimized = nullptr;
    int* input_cpu = nullptr;
    cudaMallocManaged(&input_gpu_naive, sizeof(int) * size);
    cudaMallocManaged(&input_gpu_optimized, sizeof(int) * size);
    cudaMallocManaged(&input_cpu, sizeof(int) * size);

    for (std::size_t i = 0; i < size; i++) {
            input_cpu[i] = input_gpu_optimized[i] = input_gpu_naive[i] = rand() % 10;
        }

    int sum_gpu_naive = test_reduce_naive(input_gpu_naive, size, deviceId);
    int sum_gpu_optimized = test_reduce_optimized(input_gpu_optimized, size, deviceId);
    int sum_cpu = test_cpu(input_cpu, size);

    assert(sum_cpu == sum_gpu_naive);
    assert(sum_cpu == sum_gpu_optimized);
}

