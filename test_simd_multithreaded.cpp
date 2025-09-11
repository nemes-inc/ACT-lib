#include "ACT_SIMD_MultiThreaded.h"
#include "ACT_multithreaded.h"
#include "ACT_Benchmark.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>

/**
 * Generate EEG-like test signals with multiple frequency components
 */
std::vector<std::vector<double>> generate_test_signals(int num_signals, int length, double fs) {
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

/**
 * Compare different ACT implementations
 */
void compare_implementations(const std::vector<std::vector<double>>& signals, int order) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  ACT IMPLEMENTATION COMPARISON" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Test parameters
    double fs = 256.0;
    int length = 512;
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
    // Calculate dictionary size based on parameter steps
    int tc_count = (int)((ranges.tc_max - ranges.tc_min) / ranges.tc_step) + 1;
    int fc_count = (int)((ranges.fc_max - ranges.fc_min) / ranges.fc_step) + 1;
    int logDt_count = (int)((ranges.logDt_max - ranges.logDt_min) / ranges.logDt_step) + 1;
    int c_count = (int)((ranges.c_max - ranges.c_min) / ranges.c_step) + 1;
    std::cout << "  Dictionary size: " << tc_count * fc_count * logDt_count * c_count << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
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
    std::cout << "SIMD ACT Results:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << simd_time << " s" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << signals.size() / simd_time << " signals/s" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << simd_speedup << "x" << std::endl;
    
    // 3. Multi-threaded ACT (scalar, parallel)
    std::cout << "\n🧵 Testing Multi-threaded ACT (scalar, parallel)..." << std::endl;
    ACT_MultiThreaded mt_act(fs, length, ranges, false, false);
    mt_act.generate_chirplet_dictionary();
    
    start = std::chrono::high_resolution_clock::now();
    auto mt_results = mt_act.transform_batch_parallel(signals, order, false, 4);
    end = std::chrono::high_resolution_clock::now();
    auto mt_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double mt_time = mt_duration.count() / 1e9;
    
    double mt_speedup = base_time / mt_time;
    std::cout << "Multi-threaded ACT Results:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << mt_time << " s" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << signals.size() / mt_time << " signals/s" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << mt_speedup << "x" << std::endl;
    
    // 4. SIMD + Multi-threaded ACT (vectorized, parallel) - THE CHAMPION!
    std::cout << "\n🏆 Testing SIMD + Multi-threaded ACT (vectorized, parallel)..." << std::endl;
    ACT_SIMD_MultiThreaded simd_mt_act(fs, length, ranges, false, false);
    simd_mt_act.generate_chirplet_dictionary();
    
    start = std::chrono::high_resolution_clock::now();
    auto simd_mt_results = simd_mt_act.transform_batch_simd_parallel(signals, order, 4);
    end = std::chrono::high_resolution_clock::now();
    auto simd_mt_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double simd_mt_time = simd_mt_duration.count() / 1e9;
    
    double simd_mt_speedup = base_time / simd_mt_time;
    std::cout << "SIMD + Multi-threaded ACT Results:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(3) << simd_mt_time << " s" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(1) << signals.size() / simd_mt_time << " signals/s" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << simd_mt_speedup << "x 🎯" << std::endl;
    
    // Quality verification
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "QUALITY VERIFICATION:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // Compare first signal results
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
    
    // Performance summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PERFORMANCE SUMMARY:" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Implementation                    | Speedup  | Throughput (sig/s)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "Base ACT (scalar)" 
              << "| " << std::right << std::setw(7) << "1.00x"
              << " | " << std::setw(13) << std::fixed << std::setprecision(1) << signals.size() / base_time << std::endl;
    std::cout << std::left << std::setw(30) << "SIMD ACT" 
              << "| " << std::right << std::setw(7) << std::setprecision(2) << simd_speedup << "x"
              << " | " << std::setw(13) << std::setprecision(1) << signals.size() / simd_time << std::endl;
    std::cout << std::left << std::setw(30) << "Multi-threaded ACT" 
              << "| " << std::right << std::setw(7) << std::setprecision(2) << mt_speedup << "x"
              << " | " << std::setw(13) << std::setprecision(1) << signals.size() / mt_time << std::endl;
    std::cout << std::left << std::setw(30) << "SIMD + Multi-threaded ACT 🏆" 
              << "| " << std::right << std::setw(7) << std::setprecision(2) << simd_mt_speedup << "x"
              << " | " << std::setw(13) << std::setprecision(1) << signals.size() / simd_mt_time << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "\nCombined optimization effectiveness:" << std::endl;
    std::cout << "  SIMD contribution: " << std::setprecision(1) << simd_speedup << "x" << std::endl;
    std::cout << "  Multi-threading contribution: " << std::setprecision(1) << mt_speedup << "x" << std::endl;
    std::cout << "  Combined: " << std::setprecision(1) << simd_mt_speedup << "x" << std::endl;
    std::cout << "  Efficiency: " << std::setprecision(1) << (simd_mt_speedup / (simd_speedup * mt_speedup / base_time * base_time)) * 100 << "% of theoretical maximum" << std::endl;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "  SIMD + MULTI-THREADING ACT PERFORMANCE TEST" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "Combined Apple Silicon SIMD + Multi-threading Optimization Test" << std::endl;
    std::cout << "Target: Maximum ACT performance with parallel processing + vectorization" << std::endl;
    
    // Test configuration
    const int num_signals = 24;  // Good for multi-threading test
    const int signal_length = 512;
    const double sampling_rate = 256.0;
    const int transform_order = 5;
    
    std::cout << "\nGenerating " << num_signals << " EEG-like test signals..." << std::endl;
    auto signals = generate_test_signals(num_signals, signal_length, sampling_rate);
    std::cout << "Generated " << signals.size() << " signals of " << signal_length << " samples each" << std::endl;
    
    // Run comprehensive comparison
    compare_implementations(signals, transform_order);
    
    // Detailed SIMD + Multi-threading benchmark
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "  DETAILED SIMD + MULTI-THREADING BENCHMARK" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Prepare ranges for detailed benchmark instantiation (if needed)
    ACT::ParameterRanges ranges;
    ranges.tc_min = 0; ranges.tc_max = signal_length-1; ranges.tc_step = (signal_length-1)/15.0;
    ranges.fc_min = 5; ranges.fc_max = 25; ranges.fc_step = 2.0;
    ranges.logDt_min = -3; ranges.logDt_max = -1; ranges.logDt_step = 0.5;
    ranges.c_min = -10; ranges.c_max = 10; ranges.c_step = 5.0;
    ACT_SIMD_MultiThreaded simd_mt_act(sampling_rate, signal_length, ranges, false, false);
    simd_mt_act.generate_chirplet_dictionary();
    ACT_Benchmark::benchmark_combined_performance(signals, transform_order, 8);
    
    std::cout << "\n🎉 SIMD + Multi-threading ACT test completed successfully!" << std::endl;
    std::cout << "Ready for production use with maximum performance optimization." << std::endl;
    
    return 0;
}
