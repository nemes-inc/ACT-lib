#ifndef ACT_BENCHMARK_H
#define ACT_BENCHMARK_H

#include "ACT.h"
#include "ACT_SIMD.h"
#include "ACT_multithreaded.h"
#include "ACT_SIMD_MultiThreaded.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

/**
 * Dedicated benchmarking and profiling utility for ACT implementations
 * 
 * This class provides comprehensive benchmarking functionality that was
 * previously embedded in the core ACT classes. Separating benchmarking
 * keeps the core classes focused on their primary functionality.
 */
class ACT_Benchmark {
public:
    /**
     * Benchmark different inner product implementations for SIMD ACT
     * @param simd_act SIMD ACT instance to benchmark
     * @param signal Test signal for benchmarking
     * @param iterations Number of iterations for timing
     */
    static void benchmark_simd_inner_products(ACT_SIMD& simd_act, 
                                            const std::vector<double>& signal, 
                                            int iterations = 1000);

    /**
     * Benchmark SIMD + multi-threading vs other implementations
     * @param signals Test signals
     * @param order Transform order
     * @param max_threads Maximum threads to test
     */
    static void benchmark_combined_performance(const std::vector<std::vector<double>>& signals,
                                             int order = 5, int max_threads = 8);

    /**
     * Compare all ACT implementations (base, SIMD, multi-threaded, combined)
     * @param signals Test signals
     * @param order Transform order
     * @param threads Number of threads for multi-threaded tests
     */
    static void compare_all_implementations(const std::vector<std::vector<double>>& signals,
                                          int order = 5, int threads = 4);

    /**
     * Benchmark dictionary search performance across implementations
     * @param signal Test signal
     * @param iterations Number of search iterations
     */
    static void benchmark_dictionary_search(const std::vector<double>& signal,
                                           int iterations = 100);

    /**
     * Profile memory usage and performance characteristics
     * @param signals Test signals
     * @param order Transform order
     */
    static void profile_memory_and_performance(const std::vector<std::vector<double>>& signals,
                                              int order = 5);

    /**
     * Generate EEG-like test signals for benchmarking
     * @param num_signals Number of signals to generate
     * @param length Signal length in samples
     * @param fs Sampling frequency
     * @return Vector of generated test signals
     */
    static std::vector<std::vector<double>> generate_test_signals(int num_signals, 
                                                                int length, 
                                                                double fs);

private:
    /**
     * Print performance comparison table
     */
    static void print_performance_table(const std::vector<std::string>& implementations,
                                       const std::vector<double>& times,
                                       const std::vector<double>& throughputs,
                                       const std::vector<double>& speedups);

    /**
     * Verify result consistency across implementations
     */
    static bool verify_results_consistency(const std::vector<ACT::TransformResult>& results1,
                                         const std::vector<ACT::TransformResult>& results2,
                                         double tolerance = 1e-6);

    /**
     * Calculate quality metrics from transform results
     */
    static double calculate_average_error(const std::vector<ACT::TransformResult>& results);
};

#endif // ACT_BENCHMARK_H
