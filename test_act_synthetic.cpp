// Synthetic ACT test using the existing ACT API
// Generates a sum of two chirplets and verifies the transform runs and yields low error

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <random>
#include <chrono>
#include <limits>
#include <cstdlib>
#include "ACT.h"

// Generate a single Gaussian-enveloped chirp (chirplet) sample-wise
static inline double chirplet_sample(double t, double tc, double fc, double c, double dt, double amp) {
    double time_diff = t - tc;
    double gaussian = std::exp(-0.5 * std::pow(time_diff / dt, 2));
    double phase = 2.0 * M_PI * (c * time_diff * time_diff + fc * time_diff);
    return amp * gaussian * std::cos(phase);
}

// Ground truth params corresponding to generate_synthetic_signal
// Returns a vector of {tc_samples, fc_hz, logDt, c}
static std::vector<std::vector<double>> ground_truth_params(int length, double fs) {
    double tc_step_samples = std::max(1.0, std::floor(length / 64.0));
    auto tc_samples = [&](int k) { return k * tc_step_samples; };
    auto fc_val = [&](double hz_qtr) { return 0.25 * std::round(hz_qtr / 0.25); };
    auto logdt_val = [&](double logdt) { return logdt; };
    auto c_val = [&](double c) { return std::round(c); };

    std::vector<std::vector<double>> gt;
    // Centers within window: k=16 -> 64 samples, k=48 -> 192 samples
    gt.push_back({ tc_samples(16), fc_val(10.0), logdt_val(-2.0),  c_val(8.0)  });
    gt.push_back({ tc_samples(48), fc_val(6.0), logdt_val(-1.5), c_val(-6.0) });
    return gt;
}

static std::vector<double> generate_synthetic_signal(int length, double fs) {
    std::vector<double> sig(length, 0.0);
    // Align chirplet parameters to dictionary grid used in the sweep
    // tc grid step (samples): length/64 -> convert to seconds via /fs
    double tc_step_samples = std::max(1.0, std::floor(length / 64.0));
    auto tc_sec = [&](int k) { return (k * tc_step_samples) / fs; };

    // Frequency grid: 0.25 Hz increments
    auto fc_val = [&](double hz_qtr) { return 0.25 * std::round(hz_qtr / 0.25); };

    // logDt grid: choose from {-3.0, -2.0, -1.5, -1.0}
    auto dt_from_log = [&](double logdt) { return std::exp(logdt); };

    // Chirp rate grid: integer steps (1.0 resolution)
    auto c_val = [&](double c) { return std::round(c); };

    // Two chirplets on-grid (keep centers within the window)
    double tc1 = tc_sec(16);              // 16 * (length/64) -> 64 samples
    double fc1 = fc_val(10.0);            // 10.0 Hz
    double c1  = c_val(8.0);              // 8 Hz/s
    double dt1 = dt_from_log(-2.0);       // ~0.1353 s
    double a1  = 0.9;

    double tc2 = tc_sec(48);              // 48 * (length/64) -> 192 samples
    double fc2 = fc_val(6.0);             // 6.0 Hz
    double c2  = c_val(-6.0);             // -6 Hz/s
    double dt2 = dt_from_log(-1.5);       // ~0.2231 s
    double a2  = 0.7;

    for (int i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / fs;
        sig[i] += chirplet_sample(t, tc1, fc1, c1, dt1, a1);
        sig[i] += chirplet_sample(t, tc2, fc2, c2, dt2, a2);
    }
    return sig;
}

int main() {
    try {
        // Sampling and signal (align with ACT defaults for better evaluation)
        double fs = 128.0;
        int length = 256;
        auto clean = generate_synthetic_signal(length, fs);

        // Optional noiseless baseline via env var
        bool noiseless = false;
        if (const char* e = std::getenv("ACT_SYNTH_NOISELESS")) {
            if (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "TRUE") {
                noiseless = true;
            }
        }

        // Additive white Gaussian noise with controlled input SNR
        // Define target input SNR (dB) relative to clean signal
        const double target_input_snr_db = 0.0; // challenging case
        double clean_energy = 0.0;
        for (double v : clean) clean_energy += v * v;
        double clean_power = clean_energy / length;
        double noise_power = noiseless ? 0.0 : (clean_power / std::pow(10.0, target_input_snr_db / 10.0));
        double noise_std = std::sqrt(noise_power);
        std::mt19937 rng(42u);
        std::normal_distribution<double> gauss(0.0, noise_std);
        std::vector<double> signal = clean;
        if (!noiseless) {
            for (int i = 0; i < length; ++i) signal[i] += gauss(rng);
        }

        // Validation thresholds: require minimum absolute SNR and improvement over input
        const double min_abs_snr_db = 8.0;     // absolute quality threshold
        const double min_improve_db = 6.0;     // denoising/improvement threshold

        // Precompute input SNR
        double noisy_energy = 0.0;
        for (int i = 0; i < length; ++i) noisy_energy += signal[i] * signal[i];
        double input_snr_db = noiseless ? std::numeric_limits<double>::infinity()
                                        : 10.0 * std::log10((clean_energy + 1e-12) / (noisy_energy - clean_energy + 1e-12));

        // Define a sweep of increasingly finer grids and orders to evaluate compute cost vs quality
        struct Config { double tc_step; double fc_step; double logdt_min; double logdt_max; double logdt_step; double c_step; int order; };
        std::vector<Config> sweep = {
            // Original-style bounds: logDt in [-3,-1]
            { length/32.0, 1.0,   -3.0, -1.0, 0.5, 4.0, 3 },
            { length/32.0, 0.5,   -3.0, -1.0, 0.5, 2.0, 5 },
            { length/64.0, 0.5,   -3.0, -1.0, 0.5, 2.0, 6 },
            { length/64.0, 0.25,  -3.0, -1.0, 0.5, 2.0, 6 },
            { length/64.0, 0.25,  -3.0, -1.0, 0.25, 2.0, 8 },
            { length/64.0, 0.25,  -3.0, -1.0, 0.25, 1.0, 8 },
            // README-guided diagnostic: allow logDt up to 0.0 with finer steps
            { length/64.0, 0.25,  -3.0,  0.0, 0.25, 1.0, 6 },
            { length/64.0, 0.25,  -3.0,  0.0, 0.20, 1.0, 8 }
        };

        std::cout << std::fixed << std::setprecision(2);
        if (noiseless) {
            std::cout << "Input SNR:  inf dB (noiseless baseline)\n";
        } else {
            std::cout << "Input SNR:  " << input_snr_db << " dB (target " << target_input_snr_db << " dB)\n";
        }

        double best_out = -1e9, best_imp = -1e9;
        size_t best_dict = 0; int best_order = 0;
        double best_time_ms = 0.0;

        for (size_t idx = 0; idx < sweep.size(); ++idx) {
            const auto& cfg = sweep[idx];
            std::cout << "\n=== Config " << (idx+1) << "/" << sweep.size() << " ===\n";
            std::cout << "tc_step=" << cfg.tc_step << ", fc_step=" << cfg.fc_step
                      << ", logDt in [" << cfg.logdt_min << ", " << cfg.logdt_max << "] step=" << cfg.logdt_step
                      << ", c_step=" << cfg.c_step << ", order=" << cfg.order << "\n";

            ACT::ParameterRanges ranges(
                0, length, cfg.tc_step,
                2.0, 20.0, cfg.fc_step,
                cfg.logdt_min, cfg.logdt_max, cfg.logdt_step,
                -20.0, 20.0, cfg.c_step
            );

            auto t0 = std::chrono::steady_clock::now();
            ACT act(fs, length, "synth_dict_cache.bin", ranges, false, true, false);
            auto result = act.transform(signal, cfg.order, false);
            auto t1 = std::chrono::steady_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (result.params.empty()) {
                std::cerr << "FAIL: No chirplets found" << std::endl;
                return 1;
            }

            // Compute output SNR
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
            std::cout << "Elapsed: " << elapsed_ms << " ms\n";
            std::cout << "Output SNR: " << output_snr_db << " dB, Improvement: " << improvement_db << " dB\n";

            // Recovered vs Truth reporting (greedy nearest matching)
            auto gt = ground_truth_params(length, fs);
            size_t kmax = std::min<size_t>(3, std::min(result.params.size(), gt.size()));
            std::vector<bool> gt_used(gt.size(), false);
            std::cout << "Recovered vs Truth (up to " << kmax << "):\n";
            for (size_t r = 0; r < kmax; ++r) {
                // choose recovered index r (as returned order), match to nearest unused gt
                const auto &rp = result.params[r];
                int best_idx = -1; double best_dist = 1e300;
                for (size_t gti = 0; gti < gt.size(); ++gti) {
                    if (gt_used[gti]) continue;
                    double dtc = rp[0] - gt[gti][0];
                    double dfc = rp[1] - gt[gti][1];
                    double dld = rp[2] - gt[gti][2];
                    double dc  = rp[3] - gt[gti][3];
                    double dist = dtc*dtc + dfc*dfc + dld*dld + dc*dc;
                    if (dist < best_dist) { best_dist = dist; best_idx = (int)gti; }
                }
                if (best_idx >= 0) {
                    gt_used[best_idx] = true;
                    double dtc = rp[0] - gt[best_idx][0];
                    double dfc = rp[1] - gt[best_idx][1];
                    double dld = rp[2] - gt[best_idx][2];
                    double dc  = rp[3] - gt[best_idx][3];
                    double coeff = (r < result.coeffs.size() ? result.coeffs[r] : 0.0);
                    std::cout << "  Atom " << (r+1) << ": rec(tc=" << rp[0] << ", fc=" << rp[1]
                              << ", logDt=" << rp[2] << ", c=" << rp[3] << ", a~" << coeff << ") | "
                              << "gt(tc=" << gt[best_idx][0] << ", fc=" << gt[best_idx][1]
                              << ", logDt=" << gt[best_idx][2] << ", c=" << gt[best_idx][3] << ") | "
                              << "d(tc,fc,logDt,c)=(" << dtc << ", " << dfc << ", " << dld << ", " << dc << ")\n";
                }
            }

            if (output_snr_db > best_out) {
                best_out = output_snr_db;
                best_imp = improvement_db;
                best_dict = dict_size;
                best_order = cfg.order;
                best_time_ms = elapsed_ms;
            }

            if (!noiseless && output_snr_db >= min_abs_snr_db && improvement_db >= min_improve_db) {
                std::cout << "\nSynthetic ACT test passed at this configuration." << std::endl;
                return 0;
            }
        }
        if (noiseless) {
            std::cout << "\nNoiseless baseline complete. Best Output SNR=" << best_out
                      << " dB, dict_size=" << best_dict
                      << ", order=" << best_order << ", elapsed=" << best_time_ms << " ms" << std::endl;
            return 0;
        } else {
            std::cerr << "\nFAIL: SNR thresholds not met. Best Output SNR=" << best_out
                      << " dB, Best Improvement=" << best_imp << " dB, dict_size=" << best_dict
                      << ", order=" << best_order << ", elapsed=" << best_time_ms << " ms" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
