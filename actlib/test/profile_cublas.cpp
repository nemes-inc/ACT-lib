#include "ACT_cuBLAS.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <algorithm>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
    double elapsed_s() { return elapsed_ms() / 1000.0; }
};

static std::vector<float> generate_eeg_signal_f(int length, double fs, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::vector<float> signal(length, 0.0f);
    for (int i = 0; i < length; ++i) {
        float t = static_cast<float>(static_cast<double>(i) / fs);
        signal[i] += 0.5f * std::sin(2.0f * float(M_PI) * 10.0f * t);   // Alpha
        signal[i] += 0.3f * std::sin(2.0f * float(M_PI) * 20.0f * t);   // Beta
        signal[i] += 0.4f * std::sin(2.0f * float(M_PI) * 6.0f * t);    // Theta
        float chirp_rate = 2.0f;
        signal[i] += 0.2f * std::sin(2.0f * float(M_PI) * (8.0f * t + 0.5f * chirp_rate * t * t));
        signal[i] += noise(gen);
    }
    return signal;
}

static void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

static void print_timing(const std::string& op, double ms, const std::string& details = "") {
    std::cout << std::left << std::setw(30) << op << std::right << std::setw(10)
              << std::fixed << std::setprecision(2) << ms << " ms";
    if (!details.empty()) std::cout << "  (" << details << ")";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    print_separator("ACT cuBLAS PROFILING TEST");

    // Defaults (moderate sizes for quick profiling)
    double FS = 256.0;
    int SIGNAL_LENGTH = 512;
    int TRANSFORM_ORDER = 10;
    int NUM_SIGNALS = 20;

    // Optional simple args: --len N --order P --signals K
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if ((a == "--len" || a == "-l") && i + 1 < argc) SIGNAL_LENGTH = std::atoi(argv[++i]);
        else if ((a == "--order" || a == "-o") && i + 1 < argc) TRANSFORM_ORDER = std::atoi(argv[++i]);
        else if ((a == "--signals" || a == "-k") && i + 1 < argc) NUM_SIGNALS = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << (argv && argv[0] ? argv[0] : "profile_cublas")
                      << " [--len N] [--order P] [--signals K]\n";
            return 0;
        }
    }

    std::cout << "Config:\n";
    std::cout << "  FS: " << FS << " Hz\n";
    std::cout << "  Length: " << SIGNAL_LENGTH << "\n";
    std::cout << "  Order: " << TRANSFORM_ORDER << "\n";
    std::cout << "  Signals: " << NUM_SIGNALS << "\n";

    // Parameter ranges (modest grid)
    ACT_cuBLAS_f::ParameterRanges ranges(
        0, SIGNAL_LENGTH - 1, 16,   // tc
        0.5, 20.0, 1.0,             // fc
        -1.9, -0.10, 0.08,          // logDt: duration range (16 values)
        -20.0, 20.0, 1.0          // c: chirp rate (21 values)
    );

    Timer timer;

    print_separator("DICTIONARY GENERATION");
    timer.start();
    ACT_cuBLAS_f act(FS, SIGNAL_LENGTH, ranges, true);
    double init_ms = timer.elapsed_ms();
    print_timing("Initialization", init_ms);

    timer.start();
    int dict_size = act.generate_chirplet_dictionary();
    double gen_s = timer.elapsed_s();
    print_timing("Dictionary Generation", gen_s * 1000.0, std::to_string(dict_size) + " chirplets");

    print_separator("SIGNAL GENERATION");
    timer.start();
    std::vector<std::vector<float>> signals;
    signals.reserve(NUM_SIGNALS);
    for (int i = 0; i < NUM_SIGNALS; ++i) signals.emplace_back(generate_eeg_signal_f(SIGNAL_LENGTH, FS, 42 + i));
    double sig_ms = timer.elapsed_ms();
    print_timing("Signal Generation", sig_ms, std::to_string(NUM_SIGNALS) + " signals");

    print_separator("ANALYSIS");
    std::vector<double> search_ms, transform_ms, total_ms, snrs;
    search_ms.reserve(NUM_SIGNALS);
    transform_ms.reserve(NUM_SIGNALS);
    total_ms.reserve(NUM_SIGNALS);
    snrs.reserve(NUM_SIGNALS);

    for (int i = 0; i < NUM_SIGNALS; ++i) {
        std::cout << "\nSignal " << (i + 1) << "/" << NUM_SIGNALS << "\n";
        // Warmup
        act.search_dictionary(signals[i]);

        // Search timing
        timer.start();
        auto best = act.search_dictionary(signals[i]);
        double t_search = timer.elapsed_ms();
        search_ms.push_back(t_search);
        print_timing("  Dictionary Search", t_search, "best=" + std::to_string(best.first));

        // Transform timing (full, with BFGS)
        timer.start();
        auto r = act.transform(signals[i], TRANSFORM_ORDER);
        double t_transform = timer.elapsed_s();
        transform_ms.push_back(t_transform * 1000.0);
        total_ms.push_back(t_search + t_transform * 1000.0);
        print_timing("  Full Transform", t_transform * 1000.0, std::to_string(TRANSFORM_ORDER) + " chirplets");

        // SNR
        double se = 0.0, re = 0.0;
        int n = std::min<int>(r.signal.size(), r.residue.size());
        for (int j = 0; j < n; ++j) { se += r.signal[j] * r.signal[j]; re += r.residue[j] * r.residue[j]; }
        double snr = (re > 0.0 ? 10.0 * std::log10(se / re) : 0.0);
        snrs.push_back(snr);
        std::cout << "  SNR: " << std::fixed << std::setprecision(2) << snr << " dB\n";
        std::cout << "  Residual Error: " << r.error << "\n";
    }

    auto stats = [](const std::vector<double>& v) {
        double sum = 0.0, mn = v[0], mx = v[0];
        for (double x : v) { sum += x; mn = std::min(mn, x); mx = std::max(mx, x); }
        return std::tuple<double,double,double>(sum / v.size(), mn, mx);
    };

    print_separator("PERFORMANCE SUMMARY");
    auto [mean_s, min_s, max_s] = stats(search_ms);
    auto [mean_t, min_t, max_t] = stats(transform_ms);
    auto [mean_tot, min_tot, max_tot] = stats(total_ms);
    auto [mean_snr, min_snr, max_snr] = stats(snrs);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Search:    mean=" << mean_s << " ms, range=[" << min_s << ", " << max_s << "] ms\n";
    std::cout << "Transform: mean=" << mean_t << " ms, range=[" << min_t << ", " << max_t << "] ms\n";
    std::cout << "Total:     mean=" << mean_tot << " ms, range=[" << min_tot << ", " << max_tot << "] ms\n";
    std::cout << "SNR:       mean=" << mean_snr << " dB, range=[" << min_snr << ", " << max_snr << "] dB\n";

    std::cout << "\n=== PROFILING COMPLETE ===\n";
    return 0;
}
