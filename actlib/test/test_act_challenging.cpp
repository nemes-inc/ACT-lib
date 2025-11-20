/**
 * Test ACT on challenging synthetic signal from actsleepstudy
 * Three overlapping chirplets with close frequencies
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include "ACT_CPU.h"
#include "Eigen/Dense"

using namespace std;
using namespace std::chrono;

// Helper function to calculate SNR
double calculateSNR(const vector<double>& signal, const vector<double>& noise) {
    double signal_power = 0.0, noise_power = 0.0;
    for (size_t i = 0; i < signal.size(); ++i) {
        signal_power += signal[i] * signal[i];
        noise_power += noise[i] * noise[i];
    }
    return 10.0 * log10(signal_power / noise_power);
}

// Helper to print separator
void printSeparator(const string& title) {
    cout << "\n" << string(60, '=') << endl;
    cout << setw(40) << title << endl;
    cout << string(60, '=') << endl;
}

int main() {
    printSeparator("CHALLENGING CHIRPLET DECOMPOSITION TEST");
    
    // Parameters matching actsleepstudy synthetic signal
    const double FS = 256.0;
    const int EPOCH = 2;
    const int LENGTH = EPOCH * 256;  // 512 samples
    const int ORDER = 3;  // Try to recover all 3 components
    
    cout << "Test Configuration:" << endl;
    cout << "  Sampling Rate: " << FS << " Hz" << endl;
    cout << "  Signal Length: " << LENGTH << " samples (" << EPOCH << " seconds)" << endl;
    cout << "  Transform Order: " << ORDER << " components" << endl;
    
    // Ground truth parameters from syntheticsignal.py
    struct ChirpletTruth {
        double tc, fc, logDt, c;
        const char* name;
    };
    
    vector<ChirpletTruth> ground_truth = {
        {200.0, 2.0, 0.0, 5.0, "Chirplet 1 (upchirp)"},
        {60.0, 4.0, -1.0, -5.0, "Chirplet 2 (downchirp)"},
        {100.0, 3.0, 0.0, 1.0, "Chirplet 3 (slight upchirp)"}
    };
    
    cout << "\nGround Truth Chirplets:" << endl;
    for (const auto& gt : ground_truth) {
        cout << "  " << gt.name << ": tc=" << gt.tc << ", fc=" << gt.fc 
             << " Hz, logDt=" << gt.logDt << ", c=" << gt.c << " Hz/s" << endl;
    }
    
    // Generate synthetic signal
    printSeparator("SIGNAL GENERATION");
    
    vector<double> signal(LENGTH, 0.0);
    vector<double> clean_signal(LENGTH, 0.0);
    
    // Create ACT instance for signal generation
    // Need to provide parameter ranges even for signal generation
    ACT_CPU::ParameterRanges gen_ranges;
    ACT_CPU act(FS, LENGTH, gen_ranges);
    
    // Generate each chirplet component
    for (const auto& gt : ground_truth) {
        auto chirplet = act.g(gt.tc, gt.fc, gt.logDt, gt.c);
        
        // Add to signal
        for (int i = 0; i < LENGTH; ++i) {
            clean_signal[i] += chirplet[i];
        }
    }
    
    // Normalize the clean signal
    double max_amp = *max_element(clean_signal.begin(), clean_signal.end());
    double min_amp = *min_element(clean_signal.begin(), clean_signal.end());
    double scale = max(abs(max_amp), abs(min_amp));
    if (scale > 0) {
        for (auto& s : clean_signal) s /= scale;
    }
    
    // Add small amount of noise
    const double NOISE_LEVEL = 0.01;  // 1% noise
    srand(42);
    for (int i = 0; i < LENGTH; ++i) {
        double noise = NOISE_LEVEL * (2.0 * rand() / RAND_MAX - 1.0);
        signal[i] = clean_signal[i] + noise;
    }
    
    // Calculate input SNR
    vector<double> noise(LENGTH);
    for (int i = 0; i < LENGTH; ++i) {
        noise[i] = signal[i] - clean_signal[i];
    }
    double input_snr = calculateSNR(clean_signal, noise);
    cout << "Input SNR: " << fixed << setprecision(2) << input_snr << " dB" << endl;
    
    // Define search ranges with even finer resolution
    printSeparator("ACT DECOMPOSITION");
    
    ACT_CPU::ParameterRanges ranges(
        0, LENGTH, 4,       // tc: very fine grid (was 8)
        0.5, 8.0, 0.1,      // fc: very fine steps, focused range (was 0.25 step)
        -2.0, 1.0, 0.1,     // logDt: very fine steps (was 0.2)
        -8.0, 8.0, 0.25     // c: very fine chirp rate steps (was 0.5)
    );
    
    cout << "Search ranges (with very fine resolution):" << endl;
    cout << "  tc: [0, " << LENGTH << "] with step 4 (128 time points)" << endl;
    cout << "  fc: [0.5, 8.0] Hz with step 0.1 (75 frequencies)" << endl;
    cout << "  logDt: [-2.0, 1.0] with step 0.1 (30 durations)" << endl;
    cout << "  c: [-8.0, 8.0] Hz/s with step 0.25 (64 chirp rates)" << endl;
    
    // Calculate expected dictionary size
    int tc_count = (LENGTH - 0) / 4 + 1;
    int fc_count = static_cast<int>((8.0 - 0.5) / 0.1) + 1;
    int logDt_count = static_cast<int>((1.0 - (-2.0)) / 0.1) + 1;
    int c_count = static_cast<int>((8.0 - (-8.0)) / 0.25) + 1;
    int expected_dict_size = tc_count * fc_count * logDt_count * c_count;
    cout << "  Expected dictionary size: ~" << expected_dict_size << " chirplets" << endl;
    
    // Create ACT instance for decomposition
    ACT_CPU act_decomp(FS, LENGTH, ranges, true);
    
    // Generate dictionary
    cout << "\nGenerating dictionary..." << endl;
    auto dict_start = high_resolution_clock::now();
    int dict_size = act_decomp.generate_chirplet_dictionary();
    auto dict_time = duration_cast<milliseconds>(high_resolution_clock::now() - dict_start);
    cout << "Dictionary size: " << dict_size << " chirplets" << endl;
    cout << "Dictionary generation time: " << dict_time.count() << " ms" << endl;
    
    // Run ACT decomposition
    auto start_time = high_resolution_clock::now();
    
    // Convert signal to Eigen vector
    Eigen::Map<const Eigen::VectorXd> signal_eigen(signal.data(), LENGTH);
    
    // Use transform options for control
    ACT_CPU::TransformOptions options;
    options.order = ORDER;
    options.refine = true;
    
    auto result = act_decomp.transform(signal_eigen, options);
    
    cout << "\nDecomposing signal..." << endl;
    
    auto total_time = duration_cast<milliseconds>(high_resolution_clock::now() - start_time);
    
    // The result contains the approximation
    vector<double> reconstruction(result.approx.data(), result.approx.data() + LENGTH);
    
    // Calculate output SNR
    for (int i = 0; i < LENGTH; ++i) {
        noise[i] = signal[i] - reconstruction[i];
    }
    double output_snr = calculateSNR(reconstruction, noise);
    
    printSeparator("RESULTS SUMMARY");
    
    cout << "Total decomposition time: " << total_time.count() << " ms" << endl;
    cout << "Average time per component: " << total_time.count() / ORDER << " ms" << endl;
    
    cout << "\nRecovered parameters:" << endl;
    for (int i = 0; i < result.params.rows(); ++i) {
        cout << "  Component " << i+1 << ": tc=" << result.params(i, 0) 
             << ", fc=" << result.params(i, 1) << ", logDt=" << result.params(i, 2) 
             << ", c=" << result.params(i, 3) << ", |a|=" << abs(result.coeffs[i]) << endl;
    }
    
    cout << "\nSignal quality:" << endl;
    cout << "  Input SNR: " << fixed << setprecision(2) << input_snr << " dB" << endl;
    cout << "  Output SNR: " << fixed << setprecision(2) << output_snr << " dB" << endl;
    cout << "  SNR improvement: " << fixed << setprecision(2) 
         << output_snr - input_snr << " dB" << endl;
    
    // Parameter recovery analysis
    printSeparator("PARAMETER RECOVERY ANALYSIS");
    
    cout << "Attempting to match recovered components to ground truth..." << endl;
    cout << "(Based on frequency proximity)" << endl;
    
    // Simple matching based on frequency
    vector<bool> matched(ground_truth.size(), false);
    for (int i = 0; i < min((int)result.params.rows(), (int)ground_truth.size()); ++i) {
        int best_match = -1;
        double min_freq_diff = 1e9;
        
        for (size_t j = 0; j < ground_truth.size(); ++j) {
            if (!matched[j]) {
                double freq_diff = abs(result.params(i, 1) - ground_truth[j].fc);
                if (freq_diff < min_freq_diff) {
                    min_freq_diff = freq_diff;
                    best_match = j;
                }
            }
        }
        
        if (best_match >= 0 && min_freq_diff < 2.0) {  // Within 2 Hz
            matched[best_match] = true;
            const auto& gt = ground_truth[best_match];
            
            cout << "\nComponent " << i+1 << " matched to " << gt.name << ":" << endl;
            cout << "  tc error: " << result.params(i, 0) - gt.tc << " samples" << endl;
            cout << "  fc error: " << result.params(i, 1) - gt.fc << " Hz" << endl;
            cout << "  logDt error: " << result.params(i, 2) - gt.logDt << endl;
            cout << "  c error: " << result.params(i, 3) - gt.c << " Hz/s" << endl;
        }
    }
    
    // Check which ground truth components were not found
    cout << "\nRecovery summary:" << endl;
    int found_count = 0;
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        if (matched[i]) {
            cout << "  ✓ " << ground_truth[i].name << " - FOUND" << endl;
            found_count++;
        } else {
            cout << "  ✗ " << ground_truth[i].name << " - NOT FOUND" << endl;
        }
    }
    cout << "\nRecovered " << found_count << "/" << ground_truth.size() 
         << " ground truth components" << endl;
    
    return 0;
}
