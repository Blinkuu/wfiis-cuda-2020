#include <cuda_runtime.h>


#include <helper_cuda.h>
#include <helper_functions.h>
#include <random>

// For the CUDA runtime routines (prefixed with "cuda_"

constexpr std::size_t n_bins = 100;
constexpr std::size_t range = n_bins;
constexpr std::size_t input_size = 1000;

using data_t = float;
using hist_t = uint16_t;

struct random {
  static float get() {
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    return dist(mt);
  }
};

void init(data_t** input_vec, hist_t** histogram) {
	cudaMallocManaged(input_vec, input_size * sizeof(data_t));
	cudaMallocManaged(histogram, n_bins * sizeof(hist_t));

	for(std::size_t i = 0; i < n_bins; i++){
		(*histogram)[i] = 0;
	}

	for(std::size_t i = 0; i < input_size; i++){
		(*input_vec)[i] = random::get() * range;
	}
}

void print_histogram(hist_t const * const histogram) {
	for(std::size_t i = 0; i < n_bins; i++){
		printf("[%d, %d) => %d\n", i, i + 1, histogram[i]);
	}
}

void calculate_histogram(data_t const * const data, hist_t * histogram) {
	for(std::size_t i = 0; i < input_size; i++) {
		//if(histogram[static_cast<std::size_t>(data[i])] < std::numeric_limits<hist_t>::max())
			histogram[static_cast<std::size_t>(data[i])] += 1;
	}
}

__global__ void naive_histogram_kernel(data_t const * const input_vec, hist_t * const histogram)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	//__shared__ hist_t* local_hist[n_bins] = {0};
	while(i < input_size) {
		auto index = static_cast<std::size_t>(input_vec[i]);
		atomicAdd(reinterpret_cast<int*>(histogram + index), (index % 2) ? 1 << 16 : 1);
		i += stride;
	}
}

bool test(hist_t const * const histogram_cpu, hist_t const * const histogram_gpu) {
	return memcmp(histogram_cpu, histogram_gpu, n_bins * sizeof(hist_t));
}


int main(int argc, char **argv)
{
	data_t* input_vec;
	hist_t* histogram_host;
	hist_t* histogram_device;
	init(&input_vec, &histogram_host);
	init(&input_vec, &histogram_device);
	calculate_histogram(input_vec, histogram_host);

	int threads_per_block = 32;
	int blocks_per_grid = 128;  ///(input_size + threads_per_block - 1) / threads_per_block;
	naive_histogram_kernel<<<blocks_per_grid, threads_per_block>>>(input_vec, histogram_device);
	cudaError_t err = cudaGetLastError();
	cudaDeviceSynchronize();
	std::cout << ((test(histogram_device, histogram_host)) ? "false" :  "true") << "\n";
//	std::cout << "\nGPU:\n";
//	print_histogram(histogram_device);
//	std::cout << "\nCPU:\n";
//	print_histogram(histogram_host);
}
