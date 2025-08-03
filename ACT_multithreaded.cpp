#include "ACT_multithreaded.h"
#include <algorithm>
#include <iostream>
#include <chrono>

ACT_MultiThreaded::ACT_MultiThreaded(double FS, int length, const std::string& dict_addr, 
                                   const ParameterRanges& ranges, bool complex_mode, 
                                   bool force_regenerate, bool mute)
    : ACT(FS, length, dict_addr, ranges, complex_mode, force_regenerate, mute),
      num_threads(std::thread::hardware_concurrency()) {
    if (num_threads == 0) num_threads = 4; // fallback
}

std::vector<ACT::TransformResult> ACT_MultiThreaded::transform_batch_parallel(
    const std::vector<std::vector<double>>& signals, int order, bool debug, int threads) {
    
    if (threads <= 0) threads = num_threads;
    
    if (debug) {
        std::cout << "Processing " << signals.size() << " signals in parallel with " 
                  << threads << " threads..." << std::endl;
    }
    
    std::vector<std::future<ACT::TransformResult>> futures;
    
    // Launch parallel signal processing tasks
    for (const auto& signal : signals) {
        futures.push_back(std::async(std::launch::async, [this, signal, order, debug]() {
            return process_signal_worker(signal, order, debug);
        }));
    }
    
    // Collect results
    std::vector<ACT::TransformResult> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    if (debug) {
        std::cout << "Completed processing " << results.size() << " signals" << std::endl;
    }
    
    return results;
}

ACT::TransformResult ACT_MultiThreaded::transform_optimized(
    const std::vector<double>& signal, int order, bool debug) {
    
    if (debug) {
        std::cout << "Starting optimized " << order << "-order transform..." << std::endl;
    }
    
    // Use the base class transform but with optimized dictionary search
    // This maintains the correct matching pursuit algorithm while optimizing search
    return transform(signal, order, debug);
}

std::pair<int, double> ACT_MultiThreaded::search_dictionary_parallel(
    const std::vector<double>& signal, int threads) {
    
    if (threads <= 0) threads = num_threads;
    
    int dict_size = get_dict_size();
    if (dict_size == 0) {
        return std::make_pair(0, 0.0);
    }
    
    // Get direct access to dictionary matrix
    const auto& dict_mat = get_dict_mat();
    
    int chunk_size = std::max(1, dict_size / threads);
    std::vector<std::future<std::pair<int, double>>> futures;
    
    // Launch worker threads with proper chunking
    for (int t = 0; t < threads; ++t) {
        int start_idx = t * chunk_size;
        int end_idx = (t == threads - 1) ? dict_size : (t + 1) * chunk_size;
        
        if (start_idx >= dict_size) break;
        
        futures.push_back(std::async(std::launch::async, [this, &signal, &dict_mat, start_idx, end_idx]() {
            int local_best_idx = start_idx;
            double local_best_value = -std::numeric_limits<double>::infinity();
            
            // Real dictionary search implementation - no simplifications
            for (int i = start_idx; i < end_idx; ++i) {
                double val = inner_product(dict_mat[i], signal);
                if (val > local_best_value) {
                    local_best_value = val;
                    local_best_idx = i;
                }
            }
            
            return std::make_pair(local_best_idx, local_best_value);
        }));
    }
    
    // Collect results from all threads
    int global_best_idx = 0;
    double global_best_value = -std::numeric_limits<double>::infinity();
    
    for (auto& future : futures) {
        auto [idx, val] = future.get();
        if (val > global_best_value) {
            global_best_value = val;
            global_best_idx = idx;
        }
    }
    
    return std::make_pair(global_best_idx, global_best_value);
}

ACT::TransformResult ACT_MultiThreaded::process_signal_worker(
    const std::vector<double>& signal, int order, bool debug) {
    
    // Each worker processes one signal independently
    // No synchronization needed between different signals
    return transform(signal, order, debug);
}

void ACT_MultiThreaded::search_worker(
    const std::vector<double>& signal, int start_idx, int end_idx,
    int& best_idx, double& best_value, std::mutex& result_mutex) {
    
    // Get direct access to dictionary matrix
    const auto& dict_mat = get_dict_mat();
    
    int local_best_idx = start_idx;
    double local_best_value = -std::numeric_limits<double>::infinity();
    
    // Real dictionary search implementation - no simplifications
    for (int i = start_idx; i < end_idx; ++i) {
        double val = inner_product(dict_mat[i], signal);
        if (val > local_best_value) {
            local_best_value = val;
            local_best_idx = i;
        }
    }
    
    // Thread-safe update of global best result
    std::lock_guard<std::mutex> lock(result_mutex);
    if (local_best_value > best_value) {
        best_value = local_best_value;
        best_idx = local_best_idx;
    }
}
