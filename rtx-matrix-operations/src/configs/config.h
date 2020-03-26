#pragma once

namespace cuda {
	struct config {
		static constexpr std::size_t max_iterations = 10;
		static constexpr std::size_t num_elements = 2048;
		static constexpr std::size_t threads_per_block = 32;
	};
}
