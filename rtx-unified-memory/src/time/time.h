#pragma once

#include <chrono>

namespace cuda {

	struct timer {
		using timestamp = std::chrono::time_point<std::chrono::system_clock>;
		using duration = std::chrono::duration<double>;

		timer() = delete;

		static void start() { m_timestamp1 = std::chrono::system_clock::now(); }

		static void stop() {
			m_timestamp2 = std::chrono::system_clock::now();
			m_duration = m_timestamp2 - m_timestamp1;
		}

		static double read() { return m_duration.count(); }

		static timestamp m_timestamp1;
		static timestamp m_timestamp2;
		static duration m_duration;
	};

}
