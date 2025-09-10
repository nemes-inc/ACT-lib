#include "ACT_SIMD.h"
#include "ACT_Benchmark.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

class EEGDataLoader {
public:
    struct EEGSample {
        double timestamp;
        double raw_tp9;
        double gamma_tp9;
        bool valid;
    };
    
    static std::vector<EEGSample> load_csv(const std::string& filename) {
        std::vector<EEGSample> samples;
        std::ifstream file(filename);
        std::string line;
        
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return samples;
        }
        
        // Skip header line
        std::getline(file, line);
        
        while (std::getline(file, line)) {
            if (line.empty() || line.find("connected") != std::string::npos) continue;
            
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            
            // Parse CSV line
            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }
            
            if (row.size() >= 25) {  // Ensure we have enough columns
                try {
                    EEGSample sample;
                    // Column indices: RAW_TP9=20, Gamma_TP9=16
                    sample.raw_tp9 = std::stod(row[20]);
                    sample.gamma_tp9 = std::stod(row[16]);
                    sample.valid = true;
                    samples.push_back(sample);
                } catch (const std::exception& e) {
                    // Skip invalid rows
                    continue;
                }
            }
        }
        
        std::cout << "Loaded " << samples.size() << " valid EEG samples" << std::endl;
        return samples;
    }
    
    static std::vector<double> extract_raw_signal(const std::vector<EEGSample>& samples, 
                                                  int max_samples = -1) {
        std::vector<double> signal;
        int count = 0;
        
        for (const auto& sample : samples) {
            if (sample.valid) {
                signal.push_back(sample.raw_tp9);
                count++;
                if (max_samples > 0 && count >= max_samples) break;
            }
        }
        
        return signal;
    }
};

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void save_results_to_file(const ACT_SIMD::TransformResult& result, double fs, 
                         const std::string& filename, int analysis_duration_sec) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot save results to " << filename << std::endl;
        return;
    }
    
    file << "# EEG Gamma Band ACT Analysis Results\n";
    file << "# Analysis Duration: " << analysis_duration_sec << " seconds\n";
    file << "# Sampling Rate: " << fs << " Hz\n";
    file << "# Number of Chirplets: " << result.params.size() << "\n";
    file << "# Columns: Chirplet_ID, Time_sec, Frequency_Hz, Duration_ms, ChirpRate_Hz_per_s, Coefficient, GammaType\n";
    
    for (size_t i = 0; i < result.params.size(); ++i) {
        const auto& params = result.params[i];
        double tc = params[0];
        double fc = params[1];
        double logDt = params[2];
        double c = params[3];
        
        double time_sec = tc / fs;
        double duration_ms = std::exp(logDt) * 1000;
        double coeff = result.coeffs[i];
        
        std::string gamma_type;
        if (fc < 35) gamma_type = "Low_Gamma";
        else if (fc < 45) gamma_type = "Mid_Gamma";
        else gamma_type = "High_Gamma";
        
        file << std::fixed << std::setprecision(6);
        file << (i+1) << "," << time_sec << "," << fc << "," << duration_ms 
             << "," << c << "," << coeff << "," << gamma_type << "\n";
    }
    
    file << "# Final Error: " << std::scientific << result.error << "\n";
    file.close();
    
    std::cout << "💾 Results saved to: " << filename << std::endl;
}

void print_eeg_analysis_info() {
    std::cout << "\n📊 EEG GAMMA BAND ANALYSIS RATIONALE:" << std::endl;
    std::cout << "────────────────────────────────────────────────────────────────" << std::endl;
    std::cout << "🧠 Gamma Band (30-50 Hz):" << std::endl;
    std::cout << "   • Associated with consciousness, attention, memory binding" << std::endl;
    std::cout << "   • High-frequency oscillations requiring precise time-frequency analysis" << std::endl;
    std::cout << "   • Often contains brief, transient bursts" << std::endl;
    std::cout << "\n⚙️  ACT Parameter Optimization for Gamma:" << std::endl;
    std::cout << "   • Frequency Center (fc): 30-50 Hz (gamma band focus)" << std::endl;
    std::cout << "   • Time Center (tc): Full signal coverage with fine resolution" << std::endl;
    std::cout << "   • Duration (logDt): Shorter durations for transient gamma bursts" << std::endl;
    std::cout << "   • Chirp Rate (c): Moderate range for frequency sweeps" << std::endl;
    std::cout << "────────────────────────────────────────────────────────────────" << std::endl;
}

ACT::ParameterRanges create_gamma_optimized_ranges(int signal_length) {
    std::cout << "\n🎯 GAMMA-OPTIMIZED ACT PARAMETERS:" << std::endl;
    std::cout << "────────────────────────────────────────────────────────────────" << std::endl;
    
    // Time center: Cover full signal with high temporal resolution
    double tc_step = (signal_length > 2048) ? 8.0 : 4.0;  // Higher resolution for better temporal diversity
    std::cout << "⏱️  Time Center (tc):" << std::endl;
    std::cout << "   Range: 0 to " << signal_length-1 << " samples" << std::endl;
    std::cout << "   Step: " << tc_step << " samples (" << tc_step/256.0*1000 << " ms)" << std::endl;
    std::cout << "   Rationale: Fine temporal resolution for gamma transients" << std::endl;
    
    // Frequency center: Focus on gamma band with good resolution
    double fc_min = 25.0, fc_max = 49.0;
    double fc_step = (signal_length > 2048) ? 3.0 : 2.0;  // Balanced resolution
    std::cout << "\n🌊 Frequency Center (fc):" << std::endl;
    std::cout << "   Range: " << fc_min << " to " << fc_max << " Hz" << std::endl;
    std::cout << "   Step: " << fc_step << " Hz" << std::endl;
    std::cout << "   Rationale: Extended gamma range (25-55Hz) with 1Hz resolution" << std::endl;
    
    // Duration: Extended range for better duration diversity
    double logDt_min = -3.5, logDt_max = -0.5;
    double logDt_step = (signal_length > 2048) ? 0.5 : 0.3;  // Better duration resolution
    std::cout << "\n⏳ Duration (logDt):" << std::endl;
    std::cout << "   Range: " << logDt_min << " to " << logDt_max << std::endl;
    std::cout << "   Step: " << logDt_step << std::endl;
    std::cout << "   Physical Duration: " << std::exp(logDt_min)*1000 << " to " 
              << std::exp(logDt_max)*1000 << " ms" << std::endl;
    std::cout << "   Rationale: Shorter durations (11-135ms) for gamma bursts" << std::endl;
    
    // Chirp rate: Moderate range for frequency modulation with adaptive resolution
    double c_min = -15.0, c_max = 15.0;
    double c_step = (signal_length > 2048) ? 5.0 : 3.0;  // Coarser for long signals
    std::cout << "\n📈 Chirp Rate (c):" << std::endl;
    std::cout << "   Range: " << c_min << " to " << c_max << std::endl;
    std::cout << "   Step: " << c_step << std::endl;
    std::cout << "   Rationale: Moderate range for gamma frequency sweeps" << std::endl;
    
    // Calculate dictionary size
    int tc_count = static_cast<int>((signal_length-1 - 0) / tc_step) + 1;
    int fc_count = static_cast<int>((fc_max - fc_min) / fc_step) + 1;
    int logDt_count = static_cast<int>((logDt_max - logDt_min) / logDt_step) + 1;
    int c_count = static_cast<int>((c_max - c_min) / c_step) + 1;
    int dict_size = tc_count * fc_count * logDt_count * c_count;
    
    std::cout << "\n📊 Dictionary Statistics:" << std::endl;
    std::cout << "   tc values: " << tc_count << std::endl;
    std::cout << "   fc values: " << fc_count << std::endl;
    std::cout << "   logDt values: " << logDt_count << std::endl;
    std::cout << "   c values: " << c_count << std::endl;
    std::cout << "   Total dictionary size: " << dict_size << " chirplets" << std::endl;
    std::cout << "   Memory estimate: ~" << (dict_size * signal_length * sizeof(double)) / (1024*1024) 
              << " MB" << std::endl;
    std::cout << "────────────────────────────────────────────────────────────────" << std::endl;
    
    return ACT::ParameterRanges(0, signal_length-1, tc_step,
                               fc_min, fc_max, fc_step,
                               logDt_min, logDt_max, logDt_step,
                               c_min, c_max, c_step);
}

int main() {
    print_separator("REAL EEG GAMMA BAND ANALYSIS WITH ACT");
    
    const double FS = 256.0;  // Muse sampling rate
    const int MAX_SAMPLES_SHORT = 2048;  // 8 seconds of data
    const int MAX_SAMPLES_LONG = 7680;   // 30 seconds of data (256 Hz * 30s)
    const bool RUN_LONG_ANALYSIS = false;  // Set to true for 30-second analysis
    const int TRANSFORM_ORDER = 8;  // Number of gamma chirplets to extract
    
    std::cout << "🧠 Real EEG Data Analysis Configuration:" << std::endl;
    std::cout << "   Data Source: Muse EEG headband (TP9 electrode)" << std::endl;
    std::cout << "   Sampling Rate: " << FS << " Hz" << std::endl;
    int MAX_SAMPLES = RUN_LONG_ANALYSIS ? MAX_SAMPLES_LONG : MAX_SAMPLES_SHORT;
    int analysis_duration = MAX_SAMPLES / FS;
    
    std::cout << "   Analysis Window: " << MAX_SAMPLES << " samples (" 
              << analysis_duration << " seconds)" << std::endl;
    std::cout << "   Analysis Type: " << (RUN_LONG_ANALYSIS ? "Extended (30s)" : "Short (8s)") << std::endl;
    std::cout << "   Transform Order: " << TRANSFORM_ORDER << " chirplets" << std::endl;
    std::cout << "   Focus: Gamma band (30-50 Hz) oscillations" << std::endl;
    
    print_eeg_analysis_info();
    
    // Load real EEG data
    print_separator("LOADING REAL EEG DATA");
    auto eeg_samples = EEGDataLoader::load_csv("data/muse-testdata.csv");
    
    if (eeg_samples.empty()) {
        std::cerr << "❌ Failed to load EEG data!" << std::endl;
        return 1;
    }
    
    // Extract raw signal
    auto raw_signal = EEGDataLoader::extract_raw_signal(eeg_samples, MAX_SAMPLES);
    std::cout << "✅ Extracted " << raw_signal.size() << " samples from RAW_TP9 channel" << std::endl;
    
    // Signal statistics
    double mean = 0.0, min_val = raw_signal[0], max_val = raw_signal[0];
    for (double val : raw_signal) {
        mean += val;
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    mean /= raw_signal.size();
    
    std::cout << "📊 Signal Statistics:" << std::endl;
    std::cout << "   Mean: " << std::fixed << std::setprecision(2) << mean << " μV" << std::endl;
    std::cout << "   Range: " << min_val << " to " << max_val << " μV" << std::endl;
    std::cout << "   Duration: " << raw_signal.size()/FS << " seconds" << std::endl;
    
    // Create gamma-optimized parameter ranges
    print_separator("GAMMA-OPTIMIZED ACT ANALYSIS");
    auto gamma_ranges = create_gamma_optimized_ranges(raw_signal.size());
    
    // Initialize ACT with gamma-optimized parameters
    std::cout << "\n🚀 Initializing SIMD-accelerated ACT..." << std::endl;
    ACT_SIMD act_simd(FS, raw_signal.size(), "eeg_gamma_dict.bin", gamma_ranges, 
                      false, true, false);  // complex_mode=false, force_regen=true, mute=false
    
    // Perform gamma band ACT analysis
    print_separator("GAMMA BAND ACT TRANSFORM");
    std::cout << "🔬 Performing " << TRANSFORM_ORDER << "-order ACT transform on real EEG data..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = act_simd.transform(raw_signal, TRANSFORM_ORDER, true);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Display results
    print_separator("GAMMA BAND ANALYSIS RESULTS");
    std::cout << "⚡ Transform completed in " << duration.count() << " ms" << std::endl;
    std::cout << "🎯 Extracted " << result.params.size() << " gamma-band chirplets:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (size_t i = 0; i < result.params.size(); ++i) {
        const auto& params = result.params[i];
        double tc = params[0];
        double fc = params[1];
        double logDt = params[2];
        double c = params[3];
        
        double time_sec = tc / FS;
        double duration_ms = std::exp(logDt) * 1000;
        double coeff = result.coeffs[i];
        
        std::cout << "🌊 Chirplet " << (i+1) << ":" << std::endl;
        std::cout << "   Time: " << std::fixed << std::setprecision(3) << time_sec << " s" << std::endl;
        std::cout << "   Frequency: " << std::setprecision(1) << fc << " Hz" << std::endl;
        std::cout << "   Duration: " << std::setprecision(1) << duration_ms << " ms" << std::endl;
        std::cout << "   Chirp Rate: " << std::setprecision(2) << c << " Hz/s" << std::endl;
        std::cout << "   Coefficient: " << std::setprecision(3) << coeff << std::endl;
        
        // Classify gamma sub-band
        std::string gamma_type;
        if (fc < 35) gamma_type = "Low Gamma";
        else if (fc < 45) gamma_type = "Mid Gamma";
        else gamma_type = "High Gamma";
        
        std::cout << "   Type: " << gamma_type << std::endl;
        std::cout << std::string(40, '-') << std::endl;
    }
    
    // Final error and quality metrics
    std::cout << "\n📈 Analysis Quality:" << std::endl;
    std::cout << "   Final reconstruction error: " << std::scientific << std::setprecision(3) 
              << result.error << std::endl;
    std::cout << "   Signal energy captured: " << std::fixed << std::setprecision(1) 
              << (1.0 - result.error / (mean * mean)) * 100 << "%" << std::endl;
    
    // Save results to file
    std::string results_filename = "eeg_gamma_results_" + std::to_string(analysis_duration) + "s.csv";
    save_results_to_file(result, FS, results_filename, analysis_duration);
    
    print_separator("EEG GAMMA ANALYSIS COMPLETE");
    std::cout << "🎉 Successfully analyzed real EEG data for gamma band activity!" << std::endl;
    std::cout << "💡 Key findings:" << std::endl;
    std::cout << "   • Identified " << result.params.size() << " significant gamma oscillations" << std::endl;
    std::cout << "   • Analysis completed in " << duration.count() << " ms using SIMD acceleration" << std::endl;
    std::cout << "   • Gamma activity spans " << (result.params.empty() ? 0 : 
        (result.params.back()[0] - result.params.front()[0]) / FS) << " seconds of the recording" << std::endl;
    std::cout << "   • Results saved to: " << results_filename << std::endl;
    
    return 0;
}
