#include "ACT_SIMD.h"
#include "ACT_Benchmark.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

class PrecisionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        return duration.count() / 1e6;
    }
    
    double elapsed_s() {
        return elapsed_ms() / 1000.0;
    }
};

// Generate synthetic EEG-like signal
std::vector<double> generate_eeg_signal(int length, double fs, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 0.1);
    
    std::vector<double> signal(length, 0.0);
    
    for (int i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / fs;
        
        // Alpha waves (8-12 Hz)
        signal[i] += 0.5 * std::sin(2 * M_PI * 10.0 * t);
        
        // Beta waves (13-30 Hz) 
        signal[i] += 0.3 * std::sin(2 * M_PI * 20.0 * t);
        
        // Gamma waves (30-100 Hz)
        signal[i] += 0.2 * std::sin(2 * M_PI * 40.0 * t);
        
        // Add noise
        signal[i] += noise(gen);
    }
    
    return signal;
}

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void compare_act_implementations(const std::vector<double>& signal, int transform_order) {
    print_separator("ACT IMPLEMENTATION COMPARISON");
    
    // Test configuration
    const double FS = 256.0;
    const int SIGNAL_LENGTH = signal.size();
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Signal Length: " << SIGNAL_LENGTH << " samples" << std::endl;
    std::cout << "  Sampling Rate: " << FS << " Hz" << std::endl;
    std::cout << "  Transform Order: " << transform_order << " chirplets" << std::endl;
    
    // Parameter ranges for comparison
    ACT::ParameterRanges ranges(
        0, SIGNAL_LENGTH-1, 32,    // tc: 16 values
        5.0, 25.0, 2.0,           // fc: 11 values  
        -3.0, -1.0, 0.5,          // logDt: 5 values
        -10.0, 10.0, 5.0          // c: 5 values
    );
    
    PrecisionTimer timer;
    
    // Initialize base ACT
    std::cout << "\nInitializing Base ACT..." << std::endl;
    timer.start();
    ACT act_base(FS, SIGNAL_LENGTH, ranges, false, false);
    double base_init_time = timer.elapsed_s();
    std::cout << "Base ACT initialized in " << base_init_time << " s" << std::endl;
    
    // Initialize SIMD ACT
    std::cout << "\nInitializing SIMD ACT..." << std::endl;
    timer.start();
    ACT_SIMD act_simd(FS, SIGNAL_LENGTH, ranges, false, false);
    double simd_init_time = timer.elapsed_s();
    std::cout << "SIMD ACT initialized in " << simd_init_time << " s" << std::endl;
    
    // Generate dictionaries explicitly before search/transform
    act_base.generate_chirplet_dictionary();
    act_simd.generate_chirplet_dictionary();
    
    std::cout << "\nDictionary size: " << act_base.get_dict_size() << " chirplets" << std::endl;
    
    // Benchmark inner product methods
    ACT_Benchmark::benchmark_simd_inner_products(act_simd, signal, 10000);
    
    print_separator("DICTIONARY SEARCH COMPARISON");
    
    const int SEARCH_ITERATIONS = 100;
    std::cout << "Performing " << SEARCH_ITERATIONS << " dictionary searches..." << std::endl;
    
    // Benchmark base dictionary search
    timer.start();
    std::pair<int, double> base_result;
    for (int i = 0; i < SEARCH_ITERATIONS; ++i) {
        base_result = act_base.search_dictionary(signal);
    }
    double base_search_time = timer.elapsed_ms();
    
    std::cout << "\nBase ACT Dictionary Search:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << base_search_time << " ms" << std::endl;
    std::cout << "  Average per search: " << base_search_time / SEARCH_ITERATIONS << " ms" << std::endl;
    std::cout << "  Best match: index=" << base_result.first << ", value=" << std::setprecision(6) << base_result.second << std::endl;
    
    // Benchmark SIMD dictionary search
    timer.start();
    std::pair<int, double> simd_result;
    for (int i = 0; i < SEARCH_ITERATIONS; ++i) {
        simd_result = act_simd.search_dictionary(signal);
    }
    double simd_search_time = timer.elapsed_ms();
    
    std::cout << "\nSIMD ACT Dictionary Search:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << simd_search_time << " ms" << std::endl;
    std::cout << "  Average per search: " << simd_search_time / SEARCH_ITERATIONS << " ms" << std::endl;
    std::cout << "  Best match: index=" << simd_result.first << ", value=" << std::setprecision(6) << simd_result.second << std::endl;
    
    // Calculate speedup
    double speedup = base_search_time / simd_search_time;
    std::cout << "\nSIMD Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    // Verify results match
    bool results_match = (base_result.first == simd_result.first) && 
                        (std::abs(base_result.second - simd_result.second) < 1e-10);
    std::cout << "Results match: " << (results_match ? "✅ Yes" : "❌ No") << std::endl;
    
    if (!results_match) {
        std::cout << "  Difference in values: " << std::abs(base_result.second - simd_result.second) << std::endl;
    }
    
    print_separator("FULL TRANSFORM COMPARISON");
    
    // Compare full transform performance
    std::cout << "Performing full ACT transforms..." << std::endl;
    
    // Base ACT transform
    timer.start();
    auto base_transform = act_base.transform(signal, transform_order);
    double base_transform_time = timer.elapsed_ms();
    
    std::cout << "\nBase ACT Transform:" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << base_transform_time << " ms" << std::endl;
    std::cout << "  Final error: " << std::setprecision(6) << base_transform.error << std::endl;
    std::cout << "  Chirplets found: " << base_transform.params.size() << std::endl;
    
    // SIMD ACT transform
    timer.start();
    auto simd_transform = act_simd.transform(signal, transform_order);
    double simd_transform_time = timer.elapsed_ms();
    
    std::cout << "\nSIMD ACT Transform:" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << simd_transform_time << " ms" << std::endl;
    std::cout << "  Final error: " << std::setprecision(6) << simd_transform.error << std::endl;
    std::cout << "  Chirplets found: " << simd_transform.params.size() << std::endl;
    
    // Calculate overall speedup
    double transform_speedup = base_transform_time / simd_transform_time;
    std::cout << "\nOverall Transform Speedup: " << std::fixed << std::setprecision(2) 
              << transform_speedup << "x" << std::endl;
    
    // Quality comparison
    double error_diff = std::abs(base_transform.error - simd_transform.error);
    std::cout << "Error difference: " << std::scientific << std::setprecision(3) << error_diff << std::endl;
    
    print_separator("PERFORMANCE SUMMARY");
    
    std::cout << "Dictionary Search Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "Overall Transform Speedup: " << std::fixed << std::setprecision(2) << transform_speedup << "x" << std::endl;
    std::cout << "Quality Preserved: " << (error_diff < 1e-6 ? "✅ Yes" : "⚠️ Check") << std::endl;
    
    // Expected vs actual performance
    std::cout << "\nPerformance Analysis:" << std::endl;
    std::cout << "  Dictionary search was " << std::setprecision(1) 
              << (base_search_time / SEARCH_ITERATIONS) / (simd_search_time / SEARCH_ITERATIONS) 
              << "x faster with SIMD" << std::endl;
    
    double dict_search_portion = (base_search_time / SEARCH_ITERATIONS) / (base_transform_time / transform_order);
    std::cout << "  Dictionary search is ~" << std::setprecision(0) << dict_search_portion * 100 
              << "% of total transform time" << std::endl;
    
    std::cout << std::string(70, '=') << std::endl;
}

int main() {
    print_separator("SIMD ACT PERFORMANCE TEST");
    
    // Test configuration
    const double FS = 256.0;
    const int SIGNAL_LENGTH = 512;
    const int TRANSFORM_ORDER = 5;
    
    std::cout << "Apple Silicon SIMD Optimization Test" << std::endl;
    std::cout << "Target: Dictionary search acceleration using Accelerate framework and NEON" << std::endl;
    
    // Generate test signal
    std::cout << "\nGenerating EEG-like test signal..." << std::endl;
    auto test_signal = generate_eeg_signal(SIGNAL_LENGTH, FS);
    std::cout << "Generated " << test_signal.size() << " sample signal" << std::endl;
    
    // Run comparison
    compare_act_implementations(test_signal, TRANSFORM_ORDER);
    
    return 0;
}
