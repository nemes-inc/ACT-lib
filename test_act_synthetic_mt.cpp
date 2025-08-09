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
#include <thread>
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

        const double min_abs_snr_db = 12.0;
        const double min_improve_db = 11.0;

        double noisy_energy = 0.0; for (int i = 0; i < length; ++i) noisy_energy += signals[0][i] * signals[0][i];
        double input_snr_db = noiseless ? std::numeric_limits<double>::infinity()
                                        : 10.0 * std::log10((clean_energy + 1e-12) / (noisy_energy - clean_energy + 1e-12));

        struct Config { double tc_step; double fc_step; double logdt_min; double logdt_max; double logdt_step; double c_step; int order; };
        std::vector<Config> sweep = {
            { length/32.0, 1.0,   -3.0, -1.0, 0.5, 4.0, 2 },
            { length/64.0, 0.5,   -3.0, -1.0, 0.5, 2.0, 2 },
            { length/64.0, 0.25,  -3.0, -1.0, 0.25, 1.0, 2 },
            { length/64.0, 0.25,  -3.0,  0.0, 0.20, 1.0, 2 }
        };

        std::cout << std::fixed << std::setprecision(2);
        if (noiseless) std::cout << "Input SNR:  inf dB (noiseless baseline)\n";
        else std::cout << "Input SNR:  " << input_snr_db << " dB (target " << target_input_snr_db << " dB)\n";

        double best_out = -1e9, best_imp = -1e9;
        size_t best_dict = 0; int best_order = 0;
        double best_time_ms = 0.0;

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

            auto t0 = std::chrono::steady_clock::now();
            ACT_MultiThreaded act(fs, length, "synth_dict_cache_mt.bin", ranges, false, true, false);
            auto t1 = std::chrono::steady_clock::now();
            double dict_elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            auto t2 = std::chrono::steady_clock::now();
            auto results = act.transform_batch_parallel(signals, cfg.order, false, 0);
            auto t3 = std::chrono::steady_clock::now();
            double trans_elapsed_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

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
            std::cout << "Dictionary generation: " << dict_elapsed_ms << " ms\n";
            std::cout << "Transform: " << trans_elapsed_ms << " ms\n";
            std::cout << "Output SNR: " << output_snr_db << " dB, Improvement: " << improvement_db << " dB\n";

            // Recovered vs Truth reporting (greedy nearest matching) for first signal
            auto gt = ground_truth_params(length, fs);
            size_t kmax = std::min<size_t>(3, std::min(result.params.size(), gt.size()));
            std::vector<bool> gt_used(gt.size(), false);
            std::cout << "Recovered vs Truth (up to " << kmax << "):\n";
            for (size_t r = 0; r < kmax; ++r) {
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

            // Track best metrics
            if (output_snr_db > best_out) {
                best_out = output_snr_db;
                best_imp = improvement_db;
                best_dict = dict_size;
                best_order = cfg.order;
                best_time_ms = dict_elapsed_ms + trans_elapsed_ms;
            }

            bool pass = (output_snr_db >= min_abs_snr_db) && (improvement_db >= min_improve_db);
            if (pass) {
                std::cout << "\nSynthetic ACT MT test passed at this configuration." << std::endl;
                if (noiseless) {
                    std::cout << "Noiseless baseline complete. Best Output SNR=" << best_out
                              << " dB, dict_size=" << best_dict << ", order=" << best_order
                              << ", elapsed=" << best_time_ms << " ms\n";
                }
                return 0;
            }
        }

        if (noiseless) {
            std::cout << "\nNoiseless baseline complete. Best Output SNR=" << best_out
                      << " dB, dict_size=" << best_dict << ", order=" << best_order
                      << ", elapsed=" << best_time_ms << " ms" << std::endl;
            return 0;
        } else {
            std::cerr << "\nFAIL: No configuration met the SNR thresholds." << std::endl;
            if (std::isfinite(input_snr_db)) {
                std::cerr << "Best Output SNR=" << best_out << " dB, Best Improvement=" << best_imp
                          << " dB, dict_size=" << best_dict << ", order=" << best_order
                          << ", elapsed=" << best_time_ms << " ms" << std::endl;
            }
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}
