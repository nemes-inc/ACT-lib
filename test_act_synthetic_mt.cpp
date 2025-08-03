// Synthetic ACT multithreaded test (per-signal parallelism, non-SIMD)
// Generates a batch of identical signals and runs ACT_MultiThreaded over them.

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <random>
#include <chrono>
#include <limits>
#include <cstdlib>
#include "ACT_multithreaded.h"

// Generate a single Gaussian-enveloped chirp (chirplet) sample-wise
static inline double chirplet_sample(double t, double tc, double fc, double c, double dt, double amp) {
    double time_diff = t - tc;
    double gaussian = std::exp(-0.5 * std::pow(time_diff / dt, 2));
    double phase = 2.0 * M_PI * (c * time_diff * time_diff + fc * time_diff);
    return amp * gaussian * std::cos(phase);
}

static std::vector<std::vector<double>> ground_truth_params(int length, double fs) {
    double tc_step_samples = std::max(1.0, std::floor(length / 64.0));
    auto tc_samples = [&](int k) { return k * tc_step_samples; };
    auto fc_val = [&](double hz_qtr) { return 0.25 * std::round(hz_qtr / 0.25); };
    auto logdt_val = [&](double logdt) { return logdt; };
    auto c_val = [&](double c) { return std::round(c); };

    std::vector<std::vector<double>> gt;
    gt.push_back({ tc_samples(16), fc_val(10.0), logdt_val(-2.0),  c_val(8.0)  });
    gt.push_back({ tc_samples(48), fc_val(6.0), logdt_val(-1.5), c_val(-6.0) });
    return gt;
}

static std::vector<double> generate_synthetic_signal(int length, double fs) {
    std::vector<double> sig(length, 0.0);
    double tc_step_samples = std::max(1.0, std::floor(length / 64.0));
    auto tc_sec = [&](int k) { return (k * tc_step_samples) / fs; };
    auto fc_val = [&](double hz_qtr) { return 0.25 * std::round(hz_qtr / 0.25); };
    auto dt_from_log = [&](double logdt) { return std::exp(logdt); };
    auto c_val = [&](double c) { return std::round(c); };

    double tc1 = tc_sec(16); double fc1 = fc_val(10.0); double c1  = c_val(8.0);  double dt1 = dt_from_log(-2.0); double a1  = 0.9;
    double tc2 = tc_sec(48); double fc2 = fc_val(6.0);  double c2  = c_val(-6.0); double dt2 = dt_from_log(-1.5); double a2  = 0.7;

    for (int i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / fs;
        sig[i] += chirplet_sample(t, tc1, fc1, c1, dt1, a1);
        sig[i] += chirplet_sample(t, tc2, fc2, c2, dt2, a2);
    }
    return sig;
}

int main() {
    try {
        double fs = 128.0;
        int length = 256;
        auto clean = generate_synthetic_signal(length, fs);

        bool noiseless = false;
        if (const char* e = std::getenv("ACT_SYNTH_NOISELESS")) {
            if (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "TRUE") noiseless = true;
        }

        const double target_input_snr_db = 0.0;
        double clean_energy = 0.0; for (double v : clean) clean_energy += v * v;
        double clean_power = clean_energy / length;
        double noise_power = noiseless ? 0.0 : (clean_power / std::pow(10.0, target_input_snr_db / 10.0));
        double noise_std = std::sqrt(noise_power);
        std::mt19937 rng(42u);
        std::normal_distribution<double> gauss(0.0, noise_std);

        // Build a batch of signals (per-signal parallelism)
        int batch = std::max(2u, std::thread::hardware_concurrency());
        std::vector<std::vector<double>> signals(batch, clean);
        if (!noiseless) {
            for (auto& s : signals) for (int i = 0; i < length; ++i) s[i] += gauss(rng);
        }

        const double min_abs_snr_db = 8.0;
        const double min_improve_db = 6.0;

        double noisy_energy = 0.0; for (int i = 0; i < length; ++i) noisy_energy += signals[0][i] * signals[0][i];
        double input_snr_db = noiseless ? std::numeric_limits<double>::infinity()
                                        : 10.0 * std::log10((clean_energy + 1e-12) / (noisy_energy - clean_energy + 1e-12));

        struct Config { double tc_step; double fc_step; double logdt_min; double logdt_max; double logdt_step; double c_step; int order; };
        std::vector<Config> sweep = {
            { length/32.0, 1.0,   -3.0, -1.0, 0.5, 4.0, 3 }
        };

        std::cout << std::fixed << std::setprecision(2);
        if (noiseless) std::cout << "Input SNR:  inf dB (noiseless baseline)\n";
        else std::cout << "Input SNR:  " << input_snr_db << " dB (target " << target_input_snr_db << " dB)\n";

        for (size_t idx = 0; idx < sweep.size(); ++idx) {
            const auto& cfg = sweep[idx];
            std::cout << "\n=== MT Config " << (idx+1) << "/" << sweep.size() << " ===\n";
            std::cout << "tc_step=" << cfg.tc_step << ", fc_step=" << cfg.fc_step
                      << ", logDt in [" << cfg.logdt_min << ", " << cfg.logdt_max << "] step=" << cfg.logdt_step
                      << ", c_step=" << cfg.c_step << ", order=" << cfg.order
                      << ", batch=" << batch << "\n";

            ACT::ParameterRanges ranges(
                0, length, cfg.tc_step,
                2.0, 20.0, cfg.fc_step,
                cfg.logdt_min, cfg.logdt_max, cfg.logdt_step,
                -20.0, 20.0, cfg.c_step
            );

            ACT_MultiThreaded act(fs, length, "synth_dict_cache_mt.bin", ranges, false, true, false);

            auto t0 = std::chrono::steady_clock::now();
            auto results = act.transform_batch_parallel(signals, cfg.order, false, 0);
            auto t1 = std::chrono::steady_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (results.empty()) { std::cerr << "FAIL: No results" << std::endl; return 1; }

            // Compute SNR for first signal (all are identical stats)
            const auto& result = results[0];
            double clean_resid_energy = 0.0;
            for (int i = 0; i < length; ++i) {
                double approx_i = result.approx[i];
                double clean_err = clean[i] - approx_i;
                clean_resid_energy += clean_err * clean_err;
            }
            double output_snr_db = 10.0 * std::log10((clean_energy + 1e-12) / (clean_resid_energy + 1e-12));
            double improvement_db = output_snr_db - input_snr_db;

            size_t dict_size = act.get_dict_size();
            double est_bytes = static_cast<double>(dict_size) * length * sizeof(double);
            double est_mb = est_bytes / (1024.0 * 1024.0);

            std::cout << "ACT dict size: " << dict_size << ", est matrix memory: ~" << est_mb << " MB\n";
            std::cout << "Batch elapsed: " << elapsed_ms << " ms (" << (elapsed_ms / batch) << " ms/signal)\n";
            std::cout << "Output SNR: " << output_snr_db << " dB, Improvement: " << improvement_db << " dB\n";

            bool pass = (output_snr_db >= min_abs_snr_db) && (improvement_db >= min_improve_db);
            if (pass) {
                std::cout << "\nSynthetic ACT MT test passed at this configuration." << std::endl;
                return 0;
            }
        }

        std::cerr << "\nFAIL: No configuration met the SNR thresholds." << std::endl;
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}
