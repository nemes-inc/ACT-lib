#include "ACT.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

/**
 * Simple test file for C++ ACT implementation
 * Tests basic functionality with small dictionary for fast execution
 */

// Generate a test signal (sum of chirplets)
std::vector<double> generate_test_signal(int length, double fs) {
    std::vector<double> signal(length, 0.0);
    
    // Create time array
    std::vector<double> t(length);
    for (int i = 0; i < length; ++i) {
        t[i] = static_cast<double>(i) / fs;
    }
    
    // Add first chirplet: center at t=0.3s, f=5Hz, chirp rate=10Hz/s
    for (int i = 0; i < length; ++i) {
        double tc = 0.3;
        double fc = 5.0;
        double c = 10.0;
        double dt = 0.1;
        
        double time_diff = t[i] - tc;
        double gaussian = std::exp(-0.5 * std::pow(time_diff / dt, 2));
        double phase = 2.0 * M_PI * (c * time_diff * time_diff + fc * time_diff);
        signal[i] += 0.8 * gaussian * std::cos(phase);
    }
    
    // Add second chirplet: center at t=0.6s, f=8Hz, chirp rate=-5Hz/s
    for (int i = 0; i < length; ++i) {
        double tc = 0.6;
        double fc = 8.0;
        double c = -5.0;
        double dt = 0.15;
        
        double time_diff = t[i] - tc;
        double gaussian = std::exp(-0.5 * std::pow(time_diff / dt, 2));
        double phase = 2.0 * M_PI * (c * time_diff * time_diff + fc * time_diff);
        signal[i] += 0.6 * gaussian * std::cos(phase);
    }
    
    // Add some noise
    for (int i = 0; i < length; ++i) {
        signal[i] += 0.05 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
    }
    
    return signal;
}

void print_vector(const std::vector<double>& vec, const std::string& name, int max_elements = 10) {
    std::cout << name << " (size=" << vec.size() << "): [";
    int n = std::min(max_elements, static_cast<int>(vec.size()));
    for (int i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < n - 1) std::cout << ", ";
    }
    if (vec.size() > max_elements) std::cout << ", ...";
    std::cout << "]\n";
}

int main() {
    std::cout << "=== C++ ACT Implementation Test ===\n\n";
    
    try {
        // Test parameters (small dictionary for fast testing)
        double fs = 64;  // Reduced sampling frequency
        int length = 32; // Reduced signal length
        
        // Create small parameter ranges for testing
        ACT::ParameterRanges test_ranges(
            0, 32, 8,      // tc: 0 to 32, step 8 (5 values)
            2, 12, 2,      // fc: 2 to 12 Hz, step 2 (5 values)  
            -3, -1, 1,     // logDt: -3 to -1, step 1 (2 values)
            -10, 10, 10    // c: -10 to 10, step 10 (2 values)
        );
        // Dictionary size: 5 × 5 × 2 × 2 = 100 chirplets
        
        std::cout << "1. Initializing ACT with small test dictionary...\n";
        ACT act(fs, length, "test_dict_cache.bin", test_ranges, false, true, false);
        
        std::cout << "\n2. Dictionary generated successfully!\n";
        std::cout << "   Dictionary size: " << act.get_dict_size() << " chirplets\n";
        std::cout << "   Signal length: " << act.get_length() << " samples\n";
        std::cout << "   Sampling frequency: " << act.get_FS() << " Hz\n";
        
        std::cout << "\n3. Testing single chirplet generation...\n";
        auto test_chirplet = act.g(16, 5.0, -2.0, 5.0);
        print_vector(test_chirplet, "Test chirplet");
        
        std::cout << "\n4. Generating test signal...\n";
        auto test_signal = generate_test_signal(length, fs);
        print_vector(test_signal, "Test signal");
        
        std::cout << "\n5. Testing dictionary search...\n";
        auto [best_idx, best_val] = act.search_dictionary(test_signal);
        std::cout << "   Best match index: " << best_idx << "\n";
        std::cout << "   Best match value: " << std::fixed << std::setprecision(6) << best_val << "\n";
        
        std::cout << "\n6. Performing 3-order ACT transform...\n";
        auto result = act.transform(test_signal, 3, true);
        
        std::cout << "\n7. Transform Results:\n";
        std::cout << "   Final error: " << std::fixed << std::setprecision(6) << result.error << "\n";
        
        std::cout << "   Chirplet parameters:\n";
        for (int i = 0; i < result.params.size(); ++i) {
            std::cout << "     Chirplet " << i+1 << ": tc=" << std::setprecision(2) << result.params[i][0]
                      << ", fc=" << result.params[i][1] 
                      << ", logDt=" << result.params[i][2]
                      << ", c=" << result.params[i][3] 
                      << ", coeff=" << std::setprecision(4) << result.coeffs[i] << "\n";
        }
        
        print_vector(result.approx, "Approximation");
        print_vector(result.residue, "Final residue");
        
        // Calculate approximation quality
        double signal_energy = 0.0, residue_energy = 0.0;
        for (int i = 0; i < length; ++i) {
            signal_energy += test_signal[i] * test_signal[i];
            residue_energy += result.residue[i] * result.residue[i];
        }
        
        double snr_db = 10.0 * std::log10(signal_energy / residue_energy);
        std::cout << "\n8. Quality Metrics:\n";
        std::cout << "   Signal energy: " << std::setprecision(4) << signal_energy << "\n";
        std::cout << "   Residue energy: " << residue_energy << "\n";
        std::cout << "   SNR: " << std::setprecision(2) << snr_db << " dB\n";
        
        std::cout << "\n=== Test completed successfully! ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
