#include "ACT_SIMD.h"
#include "ACT_Benchmark.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

/**
 * Minimal test for SIMD-accelerated optimization step
 * Tests only the minimize_this function without full ACT initialization
 */
int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "  SIMD OPTIMIZATION STEP TEST" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "Testing SIMD acceleration for optimization step (minimize_this function)" << std::endl;
    std::cout << "Target: Accelerate chirplet generation and inner product in optimization" << std::endl;
    std::cout << std::endl;

    // Test parameters
    const int signal_length = 512;
    const double sampling_freq = 256.0;
    const int num_iterations = 1000;
    
    // Create test signal
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise_dist(0.0, 0.1);
    
    std::vector<double> test_signal(signal_length);
    for (int i = 0; i < signal_length; ++i) {
        double t = static_cast<double>(i) / sampling_freq;
        // Generate a test signal with known chirplet components
        test_signal[i] = std::cos(2.0 * M_PI * (10.0 * t + 5.0 * t * t)) * std::exp(-0.5 * std::pow((t - 1.0) / 0.5, 2));
        test_signal[i] += noise_dist(gen);
    }
    
    std::cout << "Generated test signal: " << signal_length << " samples" << std::endl;
    
    // Test parameters for optimization
    std::vector<double> test_params = {256.0, 10.0, -1.0, 5.0}; // tc, fc, logDt, c
    
    try {
        // Create minimal parameter ranges for basic SIMD functionality
        ACT::ParameterRanges ranges;
        ranges.tc_min = 0; ranges.tc_max = signal_length-1; ranges.tc_step = 32;
        ranges.fc_min = 5; ranges.fc_max = 25; ranges.fc_step = 2;
        ranges.logDt_min = -3; ranges.logDt_max = -1; ranges.logDt_step = 1;
        ranges.c_min = -10; ranges.c_max = 10; ranges.c_step = 5;
        
        std::cout << "\n=== TESTING BASE ACT OPTIMIZATION ===" << std::endl;
        
        // Test base ACT minimize_this function
        ACT base_act(sampling_freq, signal_length, ranges, false, true);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        double base_result = 0.0;
        for (int i = 0; i < num_iterations; ++i) {
            base_result += base_act.minimize_this(test_params, test_signal);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto base_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Base ACT minimize_this:" << std::endl;
        std::cout << "  Time: " << base_duration.count() / 1000.0 << " ms (" << num_iterations << " iterations)" << std::endl;
        std::cout << "  Average per call: " << base_duration.count() / static_cast<double>(num_iterations) << " μs" << std::endl;
        std::cout << "  Result: " << base_result / num_iterations << std::endl;
        
        std::cout << "\n=== TESTING SIMD ACT OPTIMIZATION ===" << std::endl;
        
        // Test SIMD ACT minimize_this function
        ACT_SIMD simd_act(sampling_freq, signal_length, ranges, false, true);
        
        start_time = std::chrono::high_resolution_clock::now();
        double simd_result = 0.0;
        for (int i = 0; i < num_iterations; ++i) {
            simd_result += simd_act.minimize_this(test_params, test_signal);
        }
        end_time = std::chrono::high_resolution_clock::now();
        auto simd_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "SIMD ACT minimize_this:" << std::endl;
        std::cout << "  Time: " << simd_duration.count() / 1000.0 << " ms (" << num_iterations << " iterations)" << std::endl;
        std::cout << "  Average per call: " << simd_duration.count() / static_cast<double>(num_iterations) << " μs" << std::endl;
        std::cout << "  Result: " << simd_result / num_iterations << std::endl;
        
        // Calculate speedup
        double speedup = static_cast<double>(base_duration.count()) / static_cast<double>(simd_duration.count());
        double result_diff = std::abs(base_result - simd_result) / std::abs(base_result);
        
        std::cout << "\n=== SIMD OPTIMIZATION RESULTS ===" << std::endl;
        std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "Result accuracy: " << std::scientific << std::setprecision(3) << result_diff << " (relative difference)" << std::endl;
        
        if (speedup > 1.0 && result_diff < 1e-10) {
            std::cout << "✅ SIMD optimization step acceleration successful!" << std::endl;
        } else if (speedup > 1.0) {
            std::cout << "⚠️  SIMD optimization step faster but accuracy concerns" << std::endl;
        } else {
            std::cout << "❌ SIMD optimization step not faster than baseline" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during SIMD optimization test: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 SIMD optimization step test completed!" << std::endl;
    return 0;
}
