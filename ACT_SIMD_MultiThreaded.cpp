#include "ACT_SIMD_MultiThreaded.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

ACT_SIMD_MultiThreaded::ACT_SIMD_MultiThreaded(double FS, int length, const ParameterRanges& ranges, bool complex_mode, bool verbose)
    : ACT_SIMD(FS, length, ranges, complex_mode, verbose),
      num_threads(std::thread::hardware_concurrency()) {
    
    if (num_threads == 0) num_threads = 4; // fallback
    
    if (!verbose) {
        std::cout << "\n=== SIMD + MULTI-THREADED ACT INITIALIZATION ===" << std::endl;
        std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
        std::cout << "Using threads: " << num_threads << std::endl;
        std::cout << "SIMD acceleration: ✅ Enabled" << std::endl;
        std::cout << "=================================================" << std::endl;
    }
}

ACT_SIMD_MultiThreaded::~ACT_SIMD_MultiThreaded() {
    // Clean destructor - no performance tracking
}



std::vector<ACT::TransformResult> ACT_SIMD_MultiThreaded::transform_batch_simd_parallel(
    const std::vector<std::vector<double>>& signals, int order, int threads) {
    
    if (threads <= 0) threads = num_threads;
    
    if (verbose) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  SIMD + MULTI-THREADED BATCH PROCESSING" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Signals: " << signals.size() << std::endl;
        std::cout << "Threads: " << threads << std::endl;
        std::cout << "Order: " << order << " chirplets per signal" << std::endl;
        std::cout << "SIMD: ✅ Enabled (Apple Accelerate + NEON)" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
    }
    
    std::vector<std::future<ACT::TransformResult>> futures;
    
    // Launch parallel signal processing tasks with SIMD acceleration
    for (const auto& signal : signals) {
        futures.push_back(std::async(std::launch::async, [this, signal, order]() {
            return process_signal_simd_worker(signal, order);
        }));
    }
    
    // Collect results
    std::vector<ACT::TransformResult> results;
    results.reserve(signals.size());
    
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    if (verbose) {
        std::cout << "\nBatch Processing Complete" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }
    
    return results;
}

ACT::TransformResult ACT_SIMD_MultiThreaded::process_signal_simd_worker(
    const std::vector<double>& signal, int order) {
    
    // Use the SIMD-accelerated transform from the parent class
    // This automatically uses SIMD for dictionary search
    return transform(signal, order);
}


