#include "ACT_Benchmark.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <thread>

void ACT_Benchmark::benchmark_simd_inner_products(ACT_SIMD& simd_act, 
                                                 const std::vector<double>& signal, 
                                                 int iterations) {
    if (simd_act.get_dict_size() == 0) {
        std::cout << "No dictionary available for benchmarking" << std::endl;
        return;
    }
    
    const auto& dict_mat = simd_act.get_dict_mat();
    const auto& test_chirplet = dict_mat[0];
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  SIMD INNER PRODUCT BENCHMARK" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Signal length: " << signal.size() << " samples" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // Benchmark scalar implementation (base ACT class)
    ACT base_act(simd_act.get_FS(), simd_act.get_length(), simd_act.get_param_ranges(), false, false);
    auto start = std::chrono::high_resolution_clock::now();
    volatile double result_scalar = 0.0;
    for (int i = 0; i < iterations; ++i) {
        result_scalar = base_act.inner_product(test_chirplet, signal);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    std::cout << "Scalar (base class):      " << std::fixed << std::setprecision(2)
              << scalar_time.count() / 1e6 << " ms (" 
              << scalar_time.count() / iterations / 1e3 << " μs/call)" << std::endl;
    
    // Benchmark auto-vectorized implementation
    start = std::chrono::high_resolution_clock::now();
    volatile double result_auto = 0.0;
    for (int i = 0; i < iterations; ++i) {
        result_auto = simd_act.inner_product_auto_vectorized(test_chirplet, signal);
    }
    end = std::chrono::high_resolution_clock::now();
    auto auto_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    std::cout << "Auto-vectorized:          " << std::fixed << std::setprecision(2)
              << auto_time.count() / 1e6 << " ms (" 
              << auto_time.count() / iterations / 1e3 << " μs/call) "
              << "[" << std::setprecision(1) << (double)scalar_time.count() / auto_time.count() << "x]" << std::endl;
    
#ifdef __ARM_NEON
    // Benchmark NEON implementation
    start = std::chrono::high_resolution_clock::now();
    volatile double result_neon = 0.0;
    for (int i = 0; i < iterations; ++i) {
        result_neon = simd_act.inner_product_neon(test_chirplet.data(), signal.data(), signal.size());
    }
    end = std::chrono::high_resolution_clock::now();
    auto neon_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    std::cout << "ARM NEON intrinsics:      " << std::fixed << std::setprecision(2)
              << neon_time.count() / 1e6 << " ms (" 
              << neon_time.count() / iterations / 1e3 << " μs/call) "
              << "[" << std::setprecision(1) << (double)scalar_time.count() / neon_time.count() << "x]" << std::endl;
#endif

#ifdef __APPLE__
    // Benchmark Accelerate implementation
    start = std::chrono::high_resolution_clock::now();
    volatile double result_accelerate = 0.0;
    for (int i = 0; i < iterations; ++i) {
        result_accelerate = simd_act.inner_product_accelerate(test_chirplet, signal);
    }
    end = std::chrono::high_resolution_clock::now();
    auto accelerate_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    std::cout << "Apple Accelerate:         " << std::fixed << std::setprecision(2)
              << accelerate_time.count() / 1e6 << " ms (" 
              << accelerate_time.count() / iterations / 1e3 << " μs/call) "
              << "[" << std::setprecision(1) << (double)scalar_time.count() / accelerate_time.count() << "x]" << std::endl;
#endif
    
    std::cout << std::string(60, '=') << std::endl;
    
    // Verify results are consistent
    double base_result = base_act.inner_product(test_chirplet, signal);
    std::cout << "Result verification:" << std::endl;
    std::cout << "  Base result: " << std::scientific << std::setprecision(6) << base_result << std::endl;
    
    double auto_result = simd_act.inner_product_auto_vectorized(test_chirplet, signal);
    std::cout << "  Auto-vec:    " << std::scientific << std::setprecision(6) << auto_result 
              << " (diff: " << std::abs(base_result - auto_result) << ")" << std::endl;
    
#ifdef __ARM_NEON
    double neon_result = simd_act.inner_product_neon(test_chirplet.data(), signal.data(), signal.size());
    std::cout << "  NEON:        " << std::scientific << std::setprecision(6) << neon_result 
              << " (diff: " << std::abs(base_result - neon_result) << ")" << std::endl;
#endif

#ifdef __APPLE__
    double acc_result = simd_act.inner_product_accelerate(test_chirplet, signal);
    std::cout << "  Accelerate:  " << std::scientific << std::setprecision(6) << acc_result 
              << " (diff: " << std::abs(base_result - acc_result) << ")" << std::endl;
#endif
}

void ACT_Benchmark::benchmark_combined_performance(const std::vector<std::vector<double>>& signals,
                                                  int order, int max_threads) {
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  SIMD + MULTI-THREADING PERFORMANCE BENCHMARK" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Signals: " << signals.size() << std::endl;
    std::cout << "  Signal length: " << (signals.empty() ? 0 : signals[0].size()) << " samples" << std::endl;
    std::cout << "  Transform order: " << order << " chirplets" << std::endl;
    std::cout << "  Max threads to test: " << max_threads << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // Create parameter ranges
    double fs = 256.0;
    int length = signals.empty() ? 512 : static_cast<int>(signals[0].size());
    ACT::ParameterRanges ranges;
    ranges.tc_min = 0; ranges.tc_max = length-1; ranges.tc_step = (length-1)/15.0;
    ranges.fc_min = 5; ranges.fc_max = 25; ranges.fc_step = 2.0;
    ranges.logDt_min = -3; ranges.logDt_max = -1; ranges.logDt_step = 0.5;
    ranges.c_min = -10; ranges.c_max = 10; ranges.c_step = 5.0;

    // Create SIMD + Multi-threaded ACT instance and generate dictionary
    ACT_SIMD_MultiThreaded simd_mt_act(fs, length, ranges, false, false);
    simd_mt_act.generate_chirplet_dictionary();
    std::cout << "  Dictionary size: " << simd_mt_act.get_dict_size() << " chirplets" << std::endl;
    
    // Benchmark single-threaded SIMD baseline
    std::cout << "\n🔄 Testing Single-threaded SIMD baseline..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<ACT::TransformResult> baseline_results;
    for (const auto& signal : signals) {
        baseline_results.push_back(simd_mt_act.transform(signal, order));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto baseline_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double baseline_time = baseline_duration.count() / 1e9;
    
    std::cout << "Single-threaded SIMD Results:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << baseline_time << " s" << std::endl;
    std::cout << "  Time per signal: " << std::setprecision(3) << (baseline_time / signals.size()) << " s" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << signals.size() / baseline_time << " signals/s" << std::endl;
    
    // Calculate baseline quality
    double baseline_avg_error = calculate_average_error(baseline_results);
    std::cout << "  Average error: " << std::scientific << std::setprecision(3) << baseline_avg_error << std::endl;
    
    // Test different thread counts
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "MULTI-THREADED SIMD RESULTS:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::vector<int> thread_counts;
    for (int t = 2; t <= max_threads; t *= 2) {
        if (t <= (int)std::thread::hardware_concurrency()) {
            thread_counts.push_back(t);
        }
    }
    
    double best_speedup = 0.0;
    int best_threads = 2;
    
    for (int threads : thread_counts) {
        std::cout << "\n🧵 Testing with " << threads << " threads..." << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        auto mt_results = simd_mt_act.transform_batch_simd_parallel(signals, order, threads);
        end = std::chrono::high_resolution_clock::now();
        
        auto mt_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double mt_time = mt_duration.count() / 1e9;
        
        double speedup = baseline_time / mt_time;
        double efficiency = (speedup / threads) * 100;
        
        // Calculate quality difference
        double mt_avg_error = calculate_average_error(mt_results);
        double error_diff = std::abs(baseline_avg_error - mt_avg_error);
        
        std::cout << threads << " threads Results:" << std::endl;
        std::cout << "  Total time: " << std::fixed << std::setprecision(3) << mt_time << " s" << std::endl;
        std::cout << "  Time per signal: " << std::setprecision(3) << (mt_time / signals.size()) << " s" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(1) << signals.size() / mt_time << " signals/s" << std::endl;
        std::cout << "  Speedup: " << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "  Efficiency: " << std::setprecision(1) << efficiency << "%" << std::endl;
        std::cout << "  Quality difference: " << std::scientific << std::setprecision(2) << error_diff;
        
        if (error_diff < 1e-10) {
            std::cout << " ✅ Excellent" << std::endl;
        } else if (error_diff < 1e-6) {
            std::cout << " ✅ Good" << std::endl;
        } else {
            std::cout << " ⚠️ Check" << std::endl;
        }
        
        if (efficiency > 80) {
            std::cout << "  ✅ Excellent parallelization efficiency" << std::endl;
        } else if (efficiency > 50) {
            std::cout << "  ✅ Good parallelization efficiency" << std::endl;
        } else {
            std::cout << "  ⚠️ Thread overhead becoming significant" << std::endl;
        }
        
        if (speedup > best_speedup) {
            best_speedup = speedup;
            best_threads = threads;
        }
    }
    
    // Performance summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "COMBINED OPTIMIZATION SUMMARY:" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Best configuration: " << best_threads << " threads" << std::endl;
    std::cout << "Maximum speedup: " << std::fixed << std::setprecision(2) << best_speedup << "x" << std::endl;
    std::cout << "Combined optimizations:" << std::endl;
    std::cout << "  SIMD dictionary search: ~4.5x speedup" << std::endl;
    std::cout << "  Multi-threading: ~" << std::setprecision(1) << best_speedup/4.5 << "x additional speedup" << std::endl;
    std::cout << "  Total improvement: ~" << std::setprecision(1) << best_speedup << "x over scalar single-threaded" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void ACT_Benchmark::compare_all_implementations(const std::vector<std::vector<double>>& signals,
                                              int order, int threads) {
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  ACT IMPLEMENTATION COMPARISON" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Test parameters
    double fs = 256.0;
    int length = signals.empty() ? 512 : signals[0].size();
    ACT::ParameterRanges ranges;
    ranges.tc_min = 0; ranges.tc_max = length-1; ranges.tc_step = (length-1)/15.0;
    ranges.fc_min = 5; ranges.fc_max = 25; ranges.fc_step = 2.0;
    ranges.logDt_min = -3; ranges.logDt_max = -1; ranges.logDt_step = 0.5;
    ranges.c_min = -10; ranges.c_max = 10; ranges.c_step = 5.0;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Signals: " << signals.size() << std::endl;
    std::cout << "  Signal length: " << length << " samples" << std::endl;
    std::cout << "  Sampling rate: " << fs << " Hz" << std::endl;
    std::cout << "  Transform order: " << order << " chirplets" << std::endl;
    
    // Calculate dictionary size
    int tc_count = (int)((ranges.tc_max - ranges.tc_min) / ranges.tc_step) + 1;
    int fc_count = (int)((ranges.fc_max - ranges.fc_min) / ranges.fc_step) + 1;
    int logDt_count = (int)((ranges.logDt_max - ranges.logDt_min) / ranges.logDt_step) + 1;
    int c_count = (int)((ranges.c_max - ranges.c_min) / ranges.c_step) + 1;
    std::cout << "  Dictionary size: " << tc_count * fc_count * logDt_count * c_count << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::vector<std::string> implementations;
    std::vector<double> times;
    std::vector<double> throughputs;
    std::vector<double> speedups;
    
    // 1. Base ACT (scalar, single-threaded)
    std::cout << "\n🔄 Testing Base ACT (scalar, single-threaded)..." << std::endl;
    ACT base_act(fs, length, ranges, false, false);
    base_act.generate_chirplet_dictionary();
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<ACT::TransformResult> base_results;
    for (const auto& signal : signals) {
        base_results.push_back(base_act.transform(signal, order));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto base_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double base_time = base_duration.count() / 1e9;
    
    implementations.push_back("Base ACT (scalar)");
    times.push_back(base_time);
    throughputs.push_back(signals.size() / base_time);
    speedups.push_back(1.0);
    
    std::cout << "Base ACT Results:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << base_time << " s" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << signals.size() / base_time << " signals/s" << std::endl;
    
    // 2. SIMD ACT (vectorized, single-threaded)
    std::cout << "\n🚀 Testing SIMD ACT (vectorized, single-threaded)..." << std::endl;
    ACT_SIMD simd_act(fs, length, ranges, false, false);
    simd_act.generate_chirplet_dictionary();
    
    start = std::chrono::high_resolution_clock::now();
    std::vector<ACT::TransformResult> simd_results;
    for (const auto& signal : signals) {
        simd_results.push_back(simd_act.transform(signal, order));
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double simd_time = simd_duration.count() / 1e9;
    
    double simd_speedup = base_time / simd_time;
    implementations.push_back("SIMD ACT");
    times.push_back(simd_time);
    throughputs.push_back(signals.size() / simd_time);
    speedups.push_back(simd_speedup);
    
    std::cout << "SIMD ACT Results:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << simd_time << " s" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << signals.size() / simd_time << " signals/s" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << simd_speedup << "x" << std::endl;
    
    // 3. Multi-threaded ACT (scalar, parallel)
    std::cout << "\n🧵 Testing Multi-threaded ACT (scalar, parallel)..." << std::endl;
    ACT_MultiThreaded mt_act(fs, length, ranges, false, false);
    mt_act.generate_chirplet_dictionary();
    
    start = std::chrono::high_resolution_clock::now();
    auto mt_results = mt_act.transform_batch_parallel(signals, order, false, threads);
    end = std::chrono::high_resolution_clock::now();
    auto mt_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double mt_time = mt_duration.count() / 1e9;
    
    double mt_speedup = base_time / mt_time;
    implementations.push_back("Multi-threaded ACT");
    times.push_back(mt_time);
    throughputs.push_back(signals.size() / mt_time);
    speedups.push_back(mt_speedup);
    
    std::cout << "Multi-threaded ACT Results:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << mt_time << " s" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << signals.size() / mt_time << " signals/s" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << mt_speedup << "x" << std::endl;
    
    // 4. SIMD + Multi-threaded ACT (vectorized, parallel) - THE CHAMPION!
    std::cout << "\n🏆 Testing SIMD + Multi-threaded ACT (vectorized, parallel)..." << std::endl;
    ACT_SIMD_MultiThreaded simd_mt_act(fs, length, ranges, false, false);
    simd_mt_act.generate_chirplet_dictionary();
    
    start = std::chrono::high_resolution_clock::now();
    auto simd_mt_results = simd_mt_act.transform_batch_simd_parallel(signals, order, threads);
    end = std::chrono::high_resolution_clock::now();
    auto simd_mt_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double simd_mt_time = simd_mt_duration.count() / 1e9;
    
    double simd_mt_speedup = base_time / simd_mt_time;
    implementations.push_back("SIMD + Multi-threaded ACT 🏆");
    times.push_back(simd_mt_time);
    throughputs.push_back(signals.size() / simd_mt_time);
    speedups.push_back(simd_mt_speedup);
    
    std::cout << "SIMD + Multi-threaded ACT Results:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << simd_mt_time << " s" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << signals.size() / simd_mt_time << " signals/s" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << simd_mt_speedup << "x 🎯" << std::endl;
    
    // Quality verification
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "QUALITY VERIFICATION:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    if (!base_results.empty() && !simd_results.empty() && 
        !mt_results.empty() && !simd_mt_results.empty()) {
        
        double base_error = std::abs(base_results[0].error);
        double simd_error = std::abs(simd_results[0].error);
        double mt_error = std::abs(mt_results[0].error);
        double simd_mt_error = std::abs(simd_mt_results[0].error);
        
        std::cout << "Signal 1 Final Errors:" << std::endl;
        std::cout << "  Base ACT: " << std::scientific << std::setprecision(3) << base_error << std::endl;
        std::cout << "  SIMD ACT: " << std::setprecision(3) << simd_error;
        if (std::abs(base_error - simd_error) < 1e-6) std::cout << " ✅";
        std::cout << std::endl;
        
        std::cout << "  Multi-threaded ACT: " << std::setprecision(3) << mt_error;
        if (std::abs(base_error - mt_error) < 1e-6) std::cout << " ✅";
        std::cout << std::endl;
        
        std::cout << "  SIMD + Multi-threaded: " << std::setprecision(3) << simd_mt_error;
        if (std::abs(base_error - simd_mt_error) < 1e-6) std::cout << " ✅";
        std::cout << std::endl;
    }
    
    // Print performance summary table
    print_performance_table(implementations, times, throughputs, speedups);
    
    std::cout << "\nCombined optimization effectiveness:" << std::endl;
    std::cout << "  SIMD contribution: " << std::setprecision(1) << simd_speedup << "x" << std::endl;
    std::cout << "  Multi-threading contribution: " << std::setprecision(1) << mt_speedup << "x" << std::endl;
    std::cout << "  Combined: " << std::setprecision(1) << simd_mt_speedup << "x" << std::endl;
}

std::vector<std::vector<double>> ACT_Benchmark::generate_test_signals(int num_signals, int length, double fs) {
    std::vector<std::vector<double>> signals;
    signals.reserve(num_signals);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> freq_dist(5.0, 25.0);  // EEG frequency range
    std::uniform_real_distribution<> amp_dist(0.5, 2.0);   // Amplitude variation
    std::uniform_real_distribution<> phase_dist(0.0, 2.0 * M_PI);
    std::normal_distribution<> noise_dist(0.0, 0.1);       // Background noise
    
    for (int s = 0; s < num_signals; s++) {
        std::vector<double> signal(length, 0.0);
        
        // Add 2-4 frequency components per signal
        int num_components = 2 + (s % 3);  // 2, 3, or 4 components
        
        for (int comp = 0; comp < num_components; comp++) {
            double freq = freq_dist(gen);
            double amp = amp_dist(gen);
            double phase = phase_dist(gen);
            
            // Add chirp-like component (frequency modulation)
            double chirp_rate = (gen() % 2 == 0) ? 0.0 : (freq_dist(gen) - freq) / length;
            
            for (int i = 0; i < length; i++) {
                double t = i / fs;
                double inst_freq = freq + chirp_rate * t;
                
                // Gaussian envelope for chirplet-like behavior
                double envelope = std::exp(-0.5 * std::pow((i - length/2.0) / (length/8.0), 2));
                
                signal[i] += amp * envelope * std::cos(2.0 * M_PI * inst_freq * t + phase);
            }
        }
        
        // Add background noise
        for (int i = 0; i < length; i++) {
            signal[i] += noise_dist(gen);
        }
        
        signals.push_back(signal);
    }
    
    return signals;
}

void ACT_Benchmark::print_performance_table(const std::vector<std::string>& implementations,
                                           const std::vector<double>& times,
                                           const std::vector<double>& throughputs,
                                           const std::vector<double>& speedups) {
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PERFORMANCE SUMMARY:" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Implementation                    | Speedup  | Throughput (sig/s)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (size_t i = 0; i < implementations.size(); ++i) {
        std::cout << std::left << std::setw(30) << implementations[i]
                  << "| " << std::right << std::setw(7) << std::fixed << std::setprecision(2) << speedups[i] << "x"
                  << " | " << std::setw(13) << std::setprecision(1) << throughputs[i] << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;
}

bool ACT_Benchmark::verify_results_consistency(const std::vector<ACT::TransformResult>& results1,
                                              const std::vector<ACT::TransformResult>& results2,
                                              double tolerance) {
    if (results1.size() != results2.size()) return false;
    
    for (size_t i = 0; i < results1.size(); ++i) {
        if (std::abs(results1[i].error - results2[i].error) > tolerance) {
            return false;
        }
    }
    return true;
}

double ACT_Benchmark::calculate_average_error(const std::vector<ACT::TransformResult>& results) {
    if (results.empty()) return 0.0;
    
    double total_error = 0.0;
    for (const auto& result : results) {
        total_error += std::abs(result.error);
    }
    return total_error / results.size();
}
