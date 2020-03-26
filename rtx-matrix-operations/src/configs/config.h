#pragma once

namespace cuda {
	struct config {
		static constexpr std::size_t max_iterations = 1;
		static constexpr std::size_t num_elements = 32;
		static constexpr std::size_t threads_per_block = 32;
	};
}
