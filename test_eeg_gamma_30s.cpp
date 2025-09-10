#include "ACT_SIMD.h"
#include "ACT_Benchmark.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

// Load EEG data from CSV file
std::vector<double> load_eeg_data(const std::string& filename, int max_samples = 7680) {
    std::vector<double> samples;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return samples;
    }
    
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line) && (max_samples <= 0 || samples.size() < max_samples)) {
        if (line.empty() || line.find("connected") != std::string::npos) continue;
        
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        
        // Parse CSV line
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        
        if (row.size() >= 21) {  // Ensure we have RAW_TP9 column (index 20)
            try {
                double raw_tp9 = std::stod(row[20]);
                samples.push_back(raw_tp9);
            } catch (const std::exception& e) {
                // Skip invalid rows
                continue;
            }
        }
    }
    
    // Remove DC offset
    if (!samples.empty()) {
        double mean = 0.0;
        for (double sample : samples) {
            mean += sample;
        }
        mean /= samples.size();
        
        for (double& sample : samples) {
            sample -= mean;
        }
    }
    
    return samples;
}

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void save_results_to_csv(const ACT_SIMD::TransformResult& result, double fs, 
                         const std::string& filename, const std::string& test_name) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return;
    }
    
    // Write CSV header
    file << "test_name,chirplet_id,tc_samples,tc_seconds,fc_hz,logDt,duration_ms,"
         << "chirp_rate_hz_per_s,coefficient,freq_start_hz,freq_end_hz\n";
    
    // Write chirplet data
    for (size_t i = 0; i < result.params.size(); ++i) {
        double tc_samples = result.params[i][0];
        double tc_seconds = tc_samples / fs;
        double fc_hz = result.params[i][1];
        double logDt = result.params[i][2];
        double duration_ms = 1000.0 * std::exp(logDt);
        double chirp_rate = result.params[i][3];
        double coefficient = result.coeffs[i];
        
        // Calculate frequency endpoints
        double half_duration = std::exp(logDt) / 2.0;
        double freq_start = fc_hz - chirp_rate * half_duration;
        double freq_end = fc_hz + chirp_rate * half_duration;
        
        file << test_name << "," << i << "," << tc_samples << "," << tc_seconds << ","
             << fc_hz << "," << logDt << "," << duration_ms << "," << chirp_rate << ","
             << coefficient << "," << freq_start << "," << freq_end << "\n";
    }
    
    file.close();
    std::cout << "✅ Results saved to: " << filename << std::endl;
}

// Create optimized parameter ranges for 30-second analysis (balanced memory/resolution)
ACT::ParameterRanges create_30s_parameter_ranges(int signal_length) {
    std::cout << "\n🎯 30-SECOND EEG GAMMA ANALYSIS PARAMETERS:" << std::endl;
    std::cout << "────────────────────────────────────────────────────────────────" << std::endl;
    
    // Balanced temporal resolution for 30s analysis
    double tc_step = 80.0;  // ~31.25ms resolution
    std::cout << "⏱️  Time Center (tc): 0 to " << signal_length-1 << " samples, step " << tc_step << std::endl;
    
    // Focused gamma band
    double fc_min = 25.0, fc_max = 49.0, fc_step = 3.0;
    std::cout << "🌊 Frequency Center (fc): " << fc_min << " to " << fc_max << " Hz, step " << fc_step << std::endl;
    
    // Balanced duration range to control memory
    double logDt_min = -3.0, logDt_max = -0.5, logDt_step = 0.5;
    std::cout << "⏳ Duration (logDt): " << logDt_min << " to " << logDt_max << ", step " << logDt_step << std::endl;
    std::cout << "   Physical Duration: " << std::exp(logDt_min)*1000 << " to " 
              << std::exp(logDt_max)*1000 << " ms" << std::endl;
    
    // Focused chirp rate range
    double c_min = -15.0, c_max = 15.0, c_step = 5.0;
    std::cout << "📈 Chirp Rate (c): " << c_min << " to " << c_max << " Hz/s, step " << c_step << std::endl;
    
    // Calculate dictionary size
    int tc_count = static_cast<int>((signal_length-1) / tc_step) + 1;
    int fc_count = static_cast<int>((fc_max - fc_min) / fc_step) + 1;
    int logDt_count = static_cast<int>((logDt_max - logDt_min) / logDt_step) + 1;
    int c_count = static_cast<int>((c_max - c_min) / c_step) + 1;
    int dict_size = tc_count * fc_count * logDt_count * c_count;
    
    std::cout << "📊 Dictionary: " << dict_size << " chirplets (~" 
              << (dict_size * signal_length * sizeof(double)) / (1024*1024) << " MB)" << std::endl;
    std::cout << "────────────────────────────────────────────────────────────────" << std::endl;
    
    return ACT::ParameterRanges(0, signal_length-1, tc_step,
                               fc_min, fc_max, fc_step,
                               logDt_min, logDt_max, logDt_step,
                               c_min, c_max, c_step);
}

int main() {
    const double FS = 256.0;  // Muse sampling rate
    const int MAX_SAMPLES = 7680;  // 30 seconds
    
    print_separator("30-SECOND EEG GAMMA BAND ACT ANALYSIS");
    
    // Load EEG data
    std::cout << "Loading EEG data (30 seconds)...\n";
    auto samples = load_eeg_data("data/muse-testdata.csv", MAX_SAMPLES);
    std::cout << "✅ Loaded " << samples.size() << " samples (" 
              << samples.size()/FS << " seconds)\n";
    
    if (samples.empty()) {
        std::cerr << "❌ No valid EEG data found!" << std::endl;
        return 1;
    }
    
    // Print signal statistics
    double min_val = *std::min_element(samples.begin(), samples.end());
    double max_val = *std::max_element(samples.begin(), samples.end());
    std::cout << "📊 Signal range: " << min_val << " to " << max_val << " μV" << std::endl;
    
    // Create parameter ranges
    auto ranges = create_30s_parameter_ranges(samples.size());
    
    // Initialize ACT with SIMD optimization
    std::cout << "\n🚀 Initializing ACT with SIMD optimization..." << std::endl;
    ACT_SIMD act_simd(FS, samples.size(), "eeg_gamma_30s_dict.bin", ranges, 
                      false, true, false);  // force regeneration for clean build
    
    // Perform ACT analysis
    std::cout << "\n🧠 Performing ACT analysis..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = act_simd.transform(samples, 5);  // Find top 5 chirplets
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "⏱️  Analysis completed in " << duration.count() << " ms" << std::endl;
    
    // Display results
    print_separator("ANALYSIS RESULTS");
    std::cout << "🔍 Detected " << result.params.size() << " gamma chirplets:" << std::endl;
    
    for (size_t i = 0; i < result.params.size(); ++i) {
        double tc_sec = result.params[i][0] / FS;
        double fc_hz = result.params[i][1];
        double duration_ms = 1000.0 * std::exp(result.params[i][2]);
        double chirp_rate = result.params[i][3];
        double coeff = result.coeffs[i];
        
        std::cout << "  Chirplet " << (i+1) << ":" << std::endl;
        std::cout << "    Time: " << std::fixed << std::setprecision(3) << tc_sec << " s" << std::endl;
        std::cout << "    Frequency: " << std::setprecision(1) << fc_hz << " Hz" << std::endl;
        std::cout << "    Duration: " << std::setprecision(0) << duration_ms << " ms" << std::endl;
        std::cout << "    Chirp Rate: " << std::setprecision(1) << chirp_rate << " Hz/s" << std::endl;
        std::cout << "    Coefficient: " << std::setprecision(4) << coeff << std::endl;
        std::cout << std::endl;
    }
    
    // Save results to CSV
    save_results_to_csv(result, FS, "eeg_gamma_results_30s.csv", "30s_analysis");
    
    std::cout << "\n🎉 30-second EEG gamma analysis complete!" << std::endl;
    std::cout << "📁 Results saved to eeg_gamma_results_30s.csv" << std::endl;
    std::cout << "🎨 Visualize with: python ../Adaptive_Chirplet_Transform/visualize_eeg_gamma.py --results original_30s" << std::endl;
    
    return 0;
}
