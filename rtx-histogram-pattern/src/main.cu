#include <cuda_runtime.h>

#include <cstdint>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <random>
#include <fstream>

// For the CUDA runtime routines (prefixed with "cuda_"

constexpr std::size_t n_bins = 2048;
constexpr std::size_t range = n_bins - 1;
constexpr std::size_t input_size = 1e9;
constexpr std::size_t optimized_threads_per_block = 1024;
constexpr std::size_t working_threads = n_bins / optimized_threads_per_block;

using data_t = float;
using hist_t = uint32_t;

struct random {
  static float get() {
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    return dist(mt);
  }
};

void init_input(data_t** input_vec) {
	cudaMallocManaged(input_vec, input_size * sizeof(data_t));

	for(std::size_t i = 0; i < input_size; i++){
		(*input_vec)[i] = random::get() * range;
	}
}

void init_histogram(hist_t** histogram) {
	cudaMallocManaged(histogram, n_bins * sizeof(hist_t));

	for(std::size_t i = 0; i < n_bins; i++){
		(*histogram)[i] = 0;
	}
}

void print_histogram(hist_t const * const histogram) {
	for(std::size_t i = 0; i < n_bins; i++){
		printf("[%d, %d) => %d\n", i, i + 1, histogram[i]);
	}
}

void calculate_histogram(data_t const * const data, hist_t * histogram) {
	for(std::size_t i = 0; i < input_size; i++) {
		if(histogram[static_cast<std::size_t>(data[i])] < std::numeric_limits<uint16_t>::max())
			histogram[static_cast<std::size_t>(data[i])] += 1;
	}
}

__global__ void naive_histogram_kernel(data_t const * const input_vec, hist_t * const histogram)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
    while(i < input_size) {
		auto index = static_cast<std::size_t>(input_vec[i]);
	    atomicAdd(histogram + index, 1);
        atomicMin(histogram + index, USHRT_MAX); // saturation handling

		i += stride;
	}
}

__global__ void shared_histogram_kernel(data_t const * const input_vec, hist_t * const histogram)
{
    __shared__ hist_t localHistogram[n_bins];

    for(std::size_t i = 0; i < n_bins; i++)
        localHistogram[i] = 0;

	__syncthreads();
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < input_size) {
        const auto index = static_cast<std::size_t>(input_vec[i]);
        atomicAdd(localHistogram + index, 1);
        atomicMin(localHistogram + index, USHRT_MAX); // saturation handling
        
        __syncthreads();

        if(threadIdx.x < working_threads) {
            int localStride = n_bins / blockDim.x;
            for(int j = threadIdx.x; j < n_bins; j += localStride) {
                atomicAdd(histogram + j, localHistogram[j]);
                atomicMin(histogram + j, USHRT_MAX);    
            }
        }
     //atomicAdd(histogram + threadIdx.x, localHistogram[threadIdx.x]);
     //atomicMin(histogram + threadIdx.x, USHRT_MAX);    
    }
}

bool test(hist_t const * const histogram_cpu, hist_t const * const histogram_gpu) {
	return memcmp(histogram_cpu, histogram_gpu, n_bins * sizeof(hist_t));
}

void run_naive_kernel(data_t const * const input_vec, hist_t * const histogram_device) {
	int threads_per_block = 32;
	int blocks_per_grid = 128;

	naive_histogram_kernel<<<blocks_per_grid, threads_per_block>>>(input_vec, histogram_device);
	cudaError_t err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << "\n";
	cudaDeviceSynchronize();
}

void run_optimized_kernel(data_t const * const input_vec, hist_t * const histogram_device) {
	int threads_per_block = optimized_threads_per_block;
	int blocks_per_grid = (input_size + threads_per_block - 1) / threads_per_block;

	shared_histogram_kernel<<<blocks_per_grid, threads_per_block>>>(input_vec, histogram_device);
	cudaError_t err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << "\n";
	cudaDeviceSynchronize();
	err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << "\n";
}

int main(int argc, char **argv)
{
	data_t* input_vec;
	hist_t* histogram_host;
	hist_t* histogram_device;
	hist_t* histogram_optimized_device;

    init_input(&input_vec);
    init_histogram(&histogram_host);
	init_histogram(&histogram_device);
	init_histogram(&histogram_optimized_device);
    calculate_histogram(input_vec, histogram_host);
    
    run_naive_kernel(input_vec, histogram_device);
    bool naive_match = test(histogram_device, histogram_host); 
    
    run_optimized_kernel(input_vec, histogram_optimized_device);
    bool optimized_match = test(histogram_optimized_device, histogram_host); 

	std::cout << ((!naive_match && !optimized_match) ? "true" : "false") << "\n";

    if (!optimized_match) {
        std::cout << "Writing to file\n";
        std::ofstream ofs;
        ofs.open("histogram.txt", std::ofstream::out);
        for(std::size_t i = 0; i < n_bins; i++) {
           ofs << i << "\t" << histogram_device[i] << "\n";
        }
        ofs.close();
    }
}
