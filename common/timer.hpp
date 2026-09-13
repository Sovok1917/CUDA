#pragma once

#include <chrono>
#include <cstdint>
#include <x86intrin.h>

/**
 * Purpose: Measures high-resolution wall-clock duration and CPU clock cycles.
 *          Uses std::chrono::high_resolution_clock and __rdtsc().
 */
class Timer {
public:
    /**
     * Purpose: Starts the cycle counter and wall-clock timer.
     * Input:   None.
     * Return:  void.
     */
    void start() {
        start_cycles_ = __rdtsc();
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    /**
     * Purpose: Stops the cycle counter and wall-clock timer.
     * Input:   None.
     * Return:  void.
     */
    void stop() {
        end_cycles_ = __rdtsc();
        end_time_ = std::chrono::high_resolution_clock::now();
    }

    /**
     * Purpose: Returns elapsed wall-clock time in milliseconds.
     * Input:   None.
     * Return:  double (elapsed time in milliseconds).
     */
    double elapsed_ms() const {
        std::chrono::duration<double, std::milli> duration = end_time_ - start_time_;
        return duration.count();
    }

    /**
     * Purpose: Returns total elapsed CPU clock cycles.
     * Input:   None.
     * Return:  uint64_t (clock cycle count).
     */
    uint64_t elapsed_cycles() const {
        return end_cycles_ - start_cycles_;
    }

private:
    uint64_t start_cycles_{0};
    uint64_t end_cycles_{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
};
