#include "ACT_CPU.h"
#include "ACT_CPU_MT.h"
#ifdef USE_MLX
#include "ACT_MLX.h"
#include "ACT_MLX_MT.h"
#endif

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstdlib>
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

// Synthetic EEG-like signal
static std::vector<double> generate_eeg_signal(int length, double fs, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 0.1);
    std::vector<double> signal(length, 0.0);
    for (int i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / fs;
        signal[i] += 0.5 * std::sin(2 * M_PI * 10.0 * t);   // alpha
        signal[i] += 0.3 * std::sin(2 * M_PI * 20.0 * t);   // beta
        signal[i] += 0.4 * std::sin(2 * M_PI * 6.0 * t);    // theta
        double chirp_rate = 2.0; // Hz/s
        signal[i] += 0.2 * std::sin(2 * M_PI * (8.0 * t + 0.5 * chirp_rate * t * t));
        signal[i] += noise(gen);
    }
    return signal;
}

static void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << std::endl;
}

static void print_timing(const std::string& op, double ms, const std::string& details = "") {
    std::cout << std::left << std::setw(30) << op
              << std::right << std::setw(10) << std::fixed << std::setprecision(2)
              << ms << " ms";
    if (!details.empty()) std::cout << "  (" << details << ")";
    std::cout << std::endl;
}

int main() {
    print_separator("C++ ACT_MT PROFILING TEST - EEG SCALE (BATCH)");

    // Configuration (mirrors profile_act.cpp defaults)
    const double FS = 256.0;         // Hz
    const int SIGNAL_LENGTH = 512;   // samples (2 s)
    const int TRANSFORM_ORDER = 10;  // chirplets

    // Batch size via env (default 16)
    int BATCH = 16;
    if (const char* b = std::getenv("ACT_PROFILE_BATCH")) {
        try { BATCH = std::max(1, std::stoi(b)); } catch (...) {}
    }
    bool coarse_only = false;
    if (const char* e = std::getenv("ACT_COARSE_ONLY")) {
        std::string s(e);
        if (s == "1" || s == "true" || s == "TRUE") coarse_only = true;
    }

    // Backend selection via env (cpu | mlx), default cpu
    bool want_mlx = false;
    if (const char* be = std::getenv("ACT_MT_BACKEND")) {
        std::string b(be);
        std::transform(b.begin(), b.end(), b.begin(), ::tolower);
        if (b == "mlx" || b == "gpu") want_mlx = true;
    }
    bool use_mlx = false;
#ifdef USE_MLX
    use_mlx = want_mlx; // only honor when compiled with MLX
#else
    (void)want_mlx;
#endif

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Sampling Rate: " << FS << " Hz" << std::endl;
    std::cout << "  Signal Length: " << SIGNAL_LENGTH << " samples (" << (SIGNAL_LENGTH/FS) << " seconds)" << std::endl;
    std::cout << "  Transform Order: " << TRANSFORM_ORDER << " chirplets" << std::endl;
    std::cout << "  Batch Size: " << BATCH << std::endl;
    std::cout << "  Coarse Only: " << (coarse_only ? 1 : 0) << std::endl;
    std::cout << "  Backend: " << (use_mlx ? "MLX_MT (float32 GEMM)" : "CPU_MT (double GEMM)") << std::endl;

    // Parameter ranges (same spirit as profile_act.cpp)
    ACT_CPU::ParameterRanges ranges(
        0, SIGNAL_LENGTH - 1, 16,   // tc
        0.5, 20.0, 1.0,             // fc
        -1.9, -0.10, 0.08,          // logDt
        -20.0, 20.0, 1.0            // c
    );

    // Expected dictionary size (approximate, step/grid inclusive)
    auto grid_count = [](double a, double b, double h) -> int {
        if (h <= 0) return 0;
        return static_cast<int>(std::floor((b - a) / h)) + 1;
    };
    int tc_count = grid_count(ranges.tc_min, ranges.tc_max, ranges.tc_step);
    int fc_count = grid_count(ranges.fc_min, ranges.fc_max, ranges.fc_step);
    int logDt_count = grid_count(ranges.logDt_min, ranges.logDt_max, ranges.logDt_step);
    int c_count = grid_count(ranges.c_min, ranges.c_max, ranges.c_step);
    int expected_dict_size = tc_count * fc_count * logDt_count * c_count;
    std::cout << "  Expected Dictionary Size: " << expected_dict_size << " chirplets" << std::endl;

    Timer timer;
    print_separator("DICTIONARY GENERATION");

    // Initialize backend and generate dictionary
    int dict_size = 0;
    double dict_memory_mb = 0.0;
    timer.start();
    if (!use_mlx) {
        ACT_CPU act(FS, SIGNAL_LENGTH, ranges, false);
        double init_time = timer.elapsed_ms();
        print_timing("CPU_MT Initialization", init_time);

        timer.start();
        dict_size = act.generate_chirplet_dictionary();
        double dict_gen_time = timer.elapsed_s();
        print_timing("Dictionary Generation", dict_gen_time * 1000.0, std::to_string(dict_size) + " chirplets");
        std::cout << "Dictionary generation rate: " << static_cast<int>(dict_size / dict_gen_time) << " chirplets/second" << std::endl;

        dict_memory_mb = (static_cast<double>(dict_size) * SIGNAL_LENGTH * sizeof(double)) / (1024.0 * 1024.0);

        print_separator("SIGNAL GENERATION AND BATCH ANALYSIS");

        // Generate batch
        std::vector<std::vector<double>> batch_signals;
        batch_signals.reserve(BATCH);
        timer.start();
        for (int i = 0; i < BATCH; ++i) batch_signals.emplace_back(generate_eeg_signal(SIGNAL_LENGTH, FS, 42 + i));
        double sig_gen_ms = timer.elapsed_ms();
        print_timing("Signal Generation", sig_gen_ms, std::to_string(BATCH) + " signals");

        // Warm-up: one dictionary search
        (void)act.search_dictionary(batch_signals[0]);

        // Prepare Eigen batch
        std::vector<Eigen::VectorXd> xs; xs.reserve(BATCH);
        for (int i = 0; i < BATCH; ++i) {
            xs.emplace_back(Eigen::Map<const Eigen::VectorXd>(batch_signals[i].data(), SIGNAL_LENGTH));
        }

        // Options
        ACT_CPU::TransformOptions opts; opts.order = TRANSFORM_ORDER; opts.residual_threshold = 1e-6; opts.refine = !coarse_only;

        // Run batch
        timer.start();
        std::vector<ACT_CPU::TransformResult> results;
        if (coarse_only) {
            results = actmt::transform_batch_gemm_coarse_only(act, xs, opts);
        } else {
            results = actmt::transform_batch(act, xs, opts, 0);
        }
        double batch_time_ms = timer.elapsed_ms();

        print_timing(coarse_only ? "Batched Coarse Transform" : "Batched Full Transform", batch_time_ms,
                     std::to_string(BATCH) + " signals");
        print_timing("Avg per-signal Transform", batch_time_ms / BATCH);

        // Compute SNRs per signal
        std::vector<double> snrs; snrs.reserve(BATCH);
        for (int i = 0; i < BATCH; ++i) {
            const auto& sig = batch_signals[i];
            const auto& res = results[i].residue;
            size_t n = std::min(sig.size(), static_cast<size_t>(res.size()));
            double se = 0.0, re = 0.0;
            for (size_t j = 0; j < n; ++j) { se += sig[j]*sig[j]; re += res[j]*res[j]; }
            double snr = (re > 0.0 ? 10.0 * std::log10(se / re) : 0.0);
            snrs.push_back(snr);
        }

        // Stats helper
        auto stats = [](const std::vector<double>& v) {
            double s = 0.0, mn = v[0], mx = v[0];
            for (double x : v) { s += x; mn = std::min(mn, x); mx = std::max(mx, x); }
            return std::make_tuple(s / v.size(), mn, mx);
        };

        print_separator("PERFORMANCE STATISTICS");
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Transform Times:" << std::endl;
        std::cout << "  Total batch: " << batch_time_ms << " ms\n";
        std::cout << "  Per-signal avg: " << (batch_time_ms / BATCH) << " ms\n";

        auto [mean_snr, min_snr, max_snr] = stats(snrs);
        std::cout << "Signal-to-Noise Ratios:" << std::endl;
        std::cout << "  Mean: " << mean_snr << " dB, Range: [" << min_snr << ", " << max_snr << "] dB\n";

        print_separator("THROUGHPUT ANALYSIS");
        double avg_analysis_time_s = (batch_time_ms / BATCH) / 1000.0;
        double signal_duration_s = SIGNAL_LENGTH / FS;
        double realtime_factor = signal_duration_s / avg_analysis_time_s;
        std::cout << "Signal Duration: " << signal_duration_s << " seconds\n";
        std::cout << "Average Analysis Time: " << avg_analysis_time_s << " seconds\n";
        std::cout << "Real-time Factor: " << std::fixed << std::setprecision(1) << realtime_factor << "x\n";
        if (realtime_factor >= 1.0) std::cout << "\u2713 Real-time processing capable!\n"; else std::cout << "\u26A0 Not real-time\n";

        print_separator("SUMMARY");
        std::cout << "Dictionary: " << dict_size << " chirplets, " << std::setprecision(1) << dict_memory_mb << " MB\n";
        std::cout << (coarse_only ? "Batched Coarse Transform" : "Batched Full Transform")
                  << ": avg " << std::setprecision(2) << (batch_time_ms / BATCH) << " ms/signal\n";
        std::cout << "Average SNR: " << std::setprecision(2) << mean_snr << " dB\n";

        std::cout << "\n=== MT PROFILING COMPLETE ===\n";
        return 0;
    }

#ifdef USE_MLX
    // MLX path (float32 backend)
    ACT_MLX_f act(FS, SIGNAL_LENGTH,
                  ACT_MLX_f::ParameterRanges(
                      ranges.tc_min, ranges.tc_max, ranges.tc_step,
                      ranges.fc_min, ranges.fc_max, ranges.fc_step,
                      ranges.logDt_min, ranges.logDt_max, ranges.logDt_step,
                      ranges.c_min, ranges.c_max, ranges.c_step
                  ),
                  false);
    double init_time = timer.elapsed_ms();
    print_timing("MLX_MT Initialization", init_time);

    timer.start();
    dict_size = act.generate_chirplet_dictionary();
    double dict_gen_time = timer.elapsed_s();
    print_timing("Dictionary Generation", dict_gen_time * 1000.0, std::to_string(dict_size) + " chirplets");
    std::cout << "Dictionary generation rate: " << static_cast<int>(dict_size / dict_gen_time) << " chirplets/second" << std::endl;

    // Memory estimate (float)
    dict_memory_mb = (static_cast<double>(dict_size) * SIGNAL_LENGTH * sizeof(float)) / (1024.0 * 1024.0);

    print_separator("SIGNAL GENERATION AND BATCH ANALYSIS");

    // Generate batch (double host vectors)
    std::vector<std::vector<double>> batch_signals;
    batch_signals.reserve(BATCH);
    timer.start();
    for (int i = 0; i < BATCH; ++i) batch_signals.emplace_back(generate_eeg_signal(SIGNAL_LENGTH, FS, 42 + i));
    double sig_gen_ms = timer.elapsed_ms();
    print_timing("Signal Generation", sig_gen_ms, std::to_string(BATCH) + " signals");

    // Prepare Eigen batch in double (the MLX_MT wrappers will cast to float internally)
    std::vector<Eigen::VectorXd> xs; xs.reserve(BATCH);
    for (int i = 0; i < BATCH; ++i) {
        xs.emplace_back(Eigen::Map<const Eigen::VectorXd>(batch_signals[i].data(), SIGNAL_LENGTH));
    }

    ACT_CPU::TransformOptions opts; opts.order = TRANSFORM_ORDER; opts.residual_threshold = 1e-6; opts.refine = !coarse_only;

    // Run batch on MLX
    timer.start();
    std::vector<ACT_MLX_f::TransformResult> results;
    if (coarse_only) {
        results = actmlx::transform_batch_mlx_gemm_coarse_only(act, xs, opts);
    } else {
        results = actmlx::transform_batch(act, xs, opts);
    }
    double batch_time_ms = timer.elapsed_ms();

    print_timing(coarse_only ? "Batched Coarse Transform" : "Batched Full Transform", batch_time_ms,
                 std::to_string(BATCH) + " signals");
    print_timing("Avg per-signal Transform", batch_time_ms / BATCH);

    // Compute SNRs per signal (residues are float)
    std::vector<double> snrs; snrs.reserve(BATCH);
    for (int i = 0; i < BATCH; ++i) {
        const auto& sig = batch_signals[i];
        const auto& res = results[i].residue;
        size_t n = std::min(sig.size(), static_cast<size_t>(res.size()));
        double se = 0.0, re = 0.0;
        for (size_t j = 0; j < n; ++j) { se += sig[j]*sig[j]; re += static_cast<double>(res[j])*static_cast<double>(res[j]); }
        double snr = (re > 0.0 ? 10.0 * std::log10(se / re) : 0.0);
        snrs.push_back(snr);
    }

    auto stats = [](const std::vector<double>& v) {
        double s = 0.0, mn = v[0], mx = v[0];
        for (double x : v) { s += x; mn = std::min(mn, x); mx = std::max(mx, x); }
        return std::make_tuple(s / v.size(), mn, mx);
    };

    print_separator("PERFORMANCE STATISTICS");
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Transform Times:" << std::endl;
    std::cout << "  Total batch: " << batch_time_ms << " ms\n";
    std::cout << "  Per-signal avg: " << (batch_time_ms / BATCH) << " ms\n";

    auto [mean_snr, min_snr, max_snr] = stats(snrs);
    std::cout << "Signal-to-Noise Ratios:" << std::endl;
    std::cout << "  Mean: " << mean_snr << " dB, Range: [" << min_snr << ", " << max_snr << "] dB\n";

    print_separator("THROUGHPUT ANALYSIS");
    double avg_analysis_time_s = (batch_time_ms / BATCH) / 1000.0;
    double signal_duration_s = SIGNAL_LENGTH / FS;
    double realtime_factor = signal_duration_s / avg_analysis_time_s;
    std::cout << "Signal Duration: " << signal_duration_s << " seconds\n";
    std::cout << "Average Analysis Time: " << avg_analysis_time_s << " seconds\n";
    std::cout << "Real-time Factor: " << std::fixed << std::setprecision(1) << realtime_factor << "x\n";
    if (realtime_factor >= 1.0) std::cout << "\u2713 Real-time processing capable!\n"; else std::cout << "\u26A0 Not real-time\n";

    print_separator("SUMMARY");
    std::cout << "Dictionary: " << dict_size << " chirplets, " << std::setprecision(1) << dict_memory_mb << " MB\n";
    std::cout << (coarse_only ? "Batched Coarse Transform" : "Batched Full Transform")
              << ": avg " << std::setprecision(2) << (batch_time_ms / BATCH) << " ms/signal\n";
    std::cout << "Average SNR: " << std::setprecision(2) << mean_snr << " dB\n";

    std::cout << "\n=== MT PROFILING COMPLETE ===\n";
    return 0;
#endif

    // (Dead code block removed)
}
