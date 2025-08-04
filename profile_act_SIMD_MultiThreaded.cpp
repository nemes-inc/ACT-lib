#include "ACT_SIMD_MultiThreaded.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // ms
    }
    double elapsed_s() { return elapsed_ms() / 1000.0; } // s
};

// Generate synthetic EEG-like signal
std::vector<double> generate_eeg_signal(int length, double fs, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 0.1);
    std::vector<double> signal(length, 0.0);
    for (int i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / fs;
        // Alpha (8-12 Hz)
        signal[i] += 0.5 * std::sin(2 * M_PI * 10.0 * t);
        // Beta (13-30 Hz)
        signal[i] += 0.3 * std::sin(2 * M_PI * 20.0 * t);
        // Theta (4-8 Hz)
        signal[i] += 0.4 * std::sin(2 * M_PI * 6.0 * t);
        // Chirp-like component
        double chirp_rate = 2.0; // Hz/s
        signal[i] += 0.2 * std::sin(2 * M_PI * (8.0 * t + 0.5 * chirp_rate * t * t));
        // Noise
        signal[i] += noise(gen);
    }
    return signal;
}

static void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << std::endl;
}

static void print_timing(const std::string& op, double time_ms, const std::string& details = "") {
    std::cout << std::left << std::setw(30) << op
              << std::right << std::setw(10) << std::fixed << std::setprecision(2)
              << time_ms << " ms";
    if (!details.empty()) std::cout << "  (" << details << ")";
    std::cout << std::endl;
}

int main() {
    print_separator("C++ ACT SIMD+MT PROFILING TEST - EEG SCALE");

    // EEG-typical parameters
    const double FS = 256.0;          // Hz
    const int SIGNAL_LENGTH = 1024;   // samples (4 s)
    const int TRANSFORM_ORDER = 10;   // chirplets per signal

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Sampling Rate: " << FS << " Hz" << std::endl;
    std::cout << "  Signal Length: " << SIGNAL_LENGTH << " samples ("
              << (SIGNAL_LENGTH/FS) << " seconds)" << std::endl;
    std::cout << "  Transform Order: " << TRANSFORM_ORDER << " chirplets" << std::endl;

    // Parameter ranges (match profile_act.cpp)
    ACT::ParameterRanges eeg_ranges(
        0, SIGNAL_LENGTH-1, 16,    // tc: step 16 (64 values)
        0.5, 50.0, 0.5,            // fc: 0.5-50 Hz (100 values)
        -4.0, -1.0, 0.2,           // logDt: (16 values)
        -20.0, 20.0, 2.0           // c: ±20 Hz/s (21 values)
    );

    // Expected dictionary size and realistic memory estimate
    int tc_count    = static_cast<int>((SIGNAL_LENGTH-1 - 0) / 16) + 1;
    int fc_count    = static_cast<int>((50.0 - 0.5) / 0.5) + 1;
    int logDt_count = static_cast<int>((-1.0 - (-4.0)) / 0.2) + 1;
    int c_count     = static_cast<int>((20.0 - (-20.0)) / 2.0) + 1;
    long long expected_dict_size = 1LL * tc_count * fc_count * logDt_count * c_count;

    double expected_memory_mb = (double)expected_dict_size * (double)SIGNAL_LENGTH * (double)sizeof(double) / (1024.0*1024.0);
    double expected_memory_gb = expected_memory_mb / 1024.0;

    std::cout << "  Expected Dictionary Size: " << expected_dict_size << " chirplets" << std::endl;
    std::cout << "  Memory Estimate: ~" << std::fixed << std::setprecision(1)
              << expected_memory_mb << " MB (" << std::setprecision(2) << expected_memory_gb << " GB)" << std::endl;

    Timer timer;

    print_separator("DICTIONARY GENERATION");

    // Initialize ACT SIMD + MT
    timer.start();
    ACT_SIMD_MultiThreaded act(FS, SIGNAL_LENGTH, "eeg_profile_dict_simd_mt.bin", eeg_ranges, false, true, false);
    double init_time = timer.elapsed_ms();
    print_timing("ACT Initialization", init_time);

    // Generate dictionary
    timer.start();
    int actual_dict_size = act.generate_chirplet_dictionary(false);
    double dict_gen_time = timer.elapsed_s();
    print_timing("Dictionary Generation", dict_gen_time * 1000.0,
                 std::to_string(actual_dict_size) + " chirplets");

    std::cout << "Dictionary generation rate: "
              << static_cast<int>(actual_dict_size / dict_gen_time) << " chirplets/second" << std::endl;

    // Realized memory usage
    double realized_memory_mb = (double)actual_dict_size * (double)SIGNAL_LENGTH * (double)sizeof(double) / (1024.0*1024.0);
    double realized_memory_gb = realized_memory_mb / 1024.0;

    print_separator("SIGNAL GENERATION AND ANALYSIS");

    // Generate multiple test signals
    const int NUM_SIGNALS = 5;
    std::vector<std::vector<double>> test_signals;

    timer.start();
    for (int i = 0; i < NUM_SIGNALS; ++i) {
        test_signals.push_back(generate_eeg_signal(SIGNAL_LENGTH, FS, 42 + i));
    }
    double signal_gen_time = timer.elapsed_ms();
    print_timing("Signal Generation", signal_gen_time, std::to_string(NUM_SIGNALS) + " signals");

    // Batch transform with SIMD + MT
    timer.start();
    auto results = act.transform_batch_simd_parallel(test_signals, TRANSFORM_ORDER, false, 0);
    double batch_time_ms = timer.elapsed_ms();

    // Compute SNRs and per-signal times (approximate per-signal = batch_time/NUM_SIGNALS)
    std::vector<double> snr_values;
    snr_values.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        double signal_energy = 0.0, residue_energy = 0.0;
        for (size_t j = 0; j < test_signals[i].size(); ++j) {
            signal_energy += test_signals[i][j] * test_signals[i][j];
            residue_energy += results[i].residue[j] * results[i].residue[j];
        }
        double snr = 10.0 * std::log10(signal_energy / residue_energy);
        snr_values.push_back(snr);
    }

    // Report batch and per-signal timing
    std::cout << "\nAnalyzed " << results.size() << " signals using SIMD + Multi-threading" << std::endl;
    print_timing("  Batch Transform", batch_time_ms, "SIMD + MT");
    print_timing("  Per-Signal (avg)", batch_time_ms / results.size());

    // Print per-signal SNRs
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "  S" << (i+1) << " SNR: " << std::fixed << std::setprecision(2) << snr_values[i] << " dB" << std::endl;
    }

    print_separator("PERFORMANCE STATISTICS");

    auto calc_stats = [](const std::vector<double>& values) {
        double sum = 0.0, min_val = values[0], max_val = values[0];
        for (double v : values) { sum += v; min_val = std::min(min_val, v); max_val = std::max(max_val, v); }
        return std::make_tuple(sum / values.size(), min_val, max_val);
    };

    auto [mean_snr, min_snr, max_snr] = calc_stats(snr_values);

    std::cout << "Dictionary Memory Usage: " << std::fixed << std::setprecision(1)
              << realized_memory_mb << " MB (" << std::setprecision(2) << realized_memory_gb << " GB)" << std::endl;

    print_separator("THROUGHPUT ANALYSIS");
    double signal_sec = (double)SIGNAL_LENGTH / FS;
    double per_signal_ms = batch_time_ms / results.size();
    double realtime_factor = signal_sec / (per_signal_ms / 1000.0);
    std::cout << "Signal Duration: " << std::fixed << std::setprecision(2) << signal_sec << " seconds" << std::endl;
    std::cout << "Average Analysis Time: " << std::fixed << std::setprecision(2) << per_signal_ms << " ms" << std::endl;
    std::cout << "Real-time Factor: " << std::fixed << std::setprecision(1) << realtime_factor << "x" << std::endl;

    print_separator("SUMMARY");
    std::cout << "Dictionary: " << actual_dict_size << " chirplets, "
              << std::fixed << std::setprecision(1) << realized_memory_mb << " MB" << std::endl;
    std::cout << "Batch Analysis: " << std::fixed << std::setprecision(2) << batch_time_ms
              << " ms (" << (per_signal_ms) << " ms/signal)" << std::endl;
    std::cout << "Average SNR: " << std::fixed << std::setprecision(2) << mean_snr << " dB" << std::endl;
    std::cout << "=== SIMD + MT PROFILING COMPLETE ===" << std::endl;

    return 0;
}
