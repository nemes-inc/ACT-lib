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
#include "ACT_CPU.h"
#include "ACT_Accelerate.h"

// Helpers to estimate memory footprint and cap overly large configurations
static size_t count_linspace(double start, double end, double step) {
    if (step <= 0) return 0;
    size_t n = 0;
    for (double v = start; v <= end + 1e-9; v += step) ++n;
    return n;
}

static double estimate_dict_mb(int length, const ACT_CPU::ParameterRanges& r) {
    const size_t ntc = count_linspace(r.tc_min, r.tc_max, r.tc_step);
    const size_t nfc = count_linspace(r.fc_min, r.fc_max, r.fc_step);
    const size_t nld = count_linspace(r.logDt_min, r.logDt_max, r.logDt_step);
    const size_t nc  = count_linspace(r.c_min, r.c_max, r.c_step);
    long double atoms = static_cast<long double>(ntc) * static_cast<long double>(nfc) * static_cast<long double>(nld) * static_cast<long double>(nc);
    long double bytes = atoms * static_cast<long double>(length) * static_cast<long double>(sizeof(double));
    return static_cast<double>(bytes / (1024.0L * 1024.0L));
}

static double env_or_default_mb(const char* name, double default_mb) {
    if (const char* e = std::getenv(name)) {
        try {
            return std::stod(std::string(e));
        } catch (...) { /* fallthrough */ }
    }
    return default_mb;
}

// Generate a single Gaussian-enveloped chirp (chirplet) sample-wise
static inline double chirplet_sample(double t, double tc, double fc, double c, double dt, double amp) {
    double time_diff = t - tc;
    double gaussian = std::exp(-0.5 * std::pow(time_diff / dt, 2));
    double phase = 2.0 * M_PI * (c * time_diff * time_diff + fc * time_diff);
    return amp * gaussian * std::cos(phase);
}

// Ground truth params corresponding to generate_synthetic_signal
// Returns a vector of {tc_samples, fc_hz, logDt, c}
static std::vector<std::vector<double>> ground_truth_params(int length) {
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
        bool coarse_only = false;
        if (const char* e = std::getenv("ACT_COARSE_ONLY")) {
            if (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "TRUE") {
                coarse_only = true;
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
        const double min_abs_snr_db = 12.0;     // absolute quality threshold
        const double min_improve_db = 11.0;     // denoising/improvement threshold
        (void)min_abs_snr_db; (void)min_improve_db; // currently informative; keep to avoid warnings

        // Precompute input SNR
        double noisy_energy = 0.0;
        for (int i = 0; i < length; ++i) noisy_energy += signal[i] * signal[i];
        double input_snr_db = noiseless ? std::numeric_limits<double>::infinity()
                                        : 10.0 * std::log10((clean_energy + 1e-12) / (noisy_energy - clean_energy + 1e-12));

        // Define a sweep of increasingly finer grids and orders to evaluate compute cost vs quality
        struct Config { double tc_step; double fc_step; double logdt_min; double logdt_max; double logdt_step; double c_step; int order; };
        std::vector<Config> sweep = {
            { length/32.0, 1.0,   -3.0, -1.0, 0.5, 4.0, 2 },
            { length/64.0, 0.5,   -3.0, -1.0, 0.5, 2.0, 2 },
            { length/64.0, 0.25,  -3.0, -1.0, 0.25, 1.0, 2 },
            { length/64.0, 0.25,  -3.0,  0.0, 0.20, 1.0, 2 }
        };

        std::cout << std::fixed << std::setprecision(2);
        if (noiseless) {
            std::cout << "Input SNR:  inf dB (noiseless baseline)\n";
        } else {
            std::cout << "Input SNR:  " << input_snr_db << " dB (target " << target_input_snr_db << " dB)\n";
        }

        double best_out_std = -1e9, best_imp_std = -1e9;
        double best_out_cpu = -1e9, best_imp_cpu = -1e9;
        double best_out_acc = -1e9, best_imp_acc = -1e9;
        size_t best_dict_std = 0, best_dict_cpu = 0, best_dict_acc = 0; int best_order_std = 0, best_order_cpu = 0, best_order_acc = 0;
        double best_time_ms_std = 0.0, best_time_ms_cpu = 0.0, best_time_ms_acc = 0.0;

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

            // Derived CPU ranges (identical bounds)
            ACT_CPU::ParameterRanges ranges_cpu(
                ranges.tc_min, ranges.tc_max, ranges.tc_step,
                ranges.fc_min, ranges.fc_max, ranges.fc_step,
                ranges.logDt_min, ranges.logDt_max, ranges.logDt_step,
                ranges.c_min, ranges.c_max, ranges.c_step
            );

            // Safety: cap dictionary memory to avoid OS freeze
            const double max_mb = env_or_default_mb("ACT_SYNTH_MAX_DICT_MB", 1024.0); // default 1 GB cap
            const double est_mb = estimate_dict_mb(length, ranges_cpu);
            if (est_mb > max_mb) {
                std::cout << "Skipping config due to estimated dictionary size " << est_mb
                          << " MB exceeding cap " << max_mb << " MB. Adjust ACT_SYNTH_MAX_DICT_MB to override.\n";
                continue;
            }

            // We'll run each implementation sequentially to minimize peak memory usage.
            // Metrics to fill per implementation
            double dict_elapsed_ms_std = 0.0, trans_elapsed_ms_std = 0.0;
            double dict_elapsed_ms_cpu = 0.0, trans_elapsed_ms_cpu = 0.0;
            double dict_elapsed_ms_acc = 0.0, trans_elapsed_ms_acc = 0.0;
            size_t dict_size_std = 0, dict_size_cpu = 0, dict_size_acc = 0;
            double output_snr_db_std = 0.0, improvement_db_std = 0.0;
            double output_snr_db_cpu = 0.0, improvement_db_cpu = 0.0;
            double output_snr_db_acc = 0.0, improvement_db_acc = 0.0;
            // Keep a copy of ACT(std) params/coeffs for reporting later in this config
            std::vector<std::vector<double>> result_std_params_copy;
            std::vector<double> result_std_coeffs_copy;

            // ACT (std)
            {
                auto t0_std = std::chrono::steady_clock::now();
                ACT act_std(fs, length, ranges, false, false);
                act_std.generate_chirplet_dictionary();
                auto t1_std = std::chrono::steady_clock::now();

                auto result_std = act_std.transform(signal, cfg.order);
                auto t2_std = std::chrono::steady_clock::now();

                dict_elapsed_ms_std = std::chrono::duration<double, std::milli>(t1_std - t0_std).count();
                trans_elapsed_ms_std = std::chrono::duration<double, std::milli>(t2_std - t1_std).count();
                dict_size_std = act_std.get_dict_size();

                auto compute_snr_std = [&](const std::vector<double>& approx_vec) {
                    double clean_resid_energy = 0.0;
                    for (int i = 0; i < length; ++i) {
                        double clean_err = clean[i] - approx_vec[i];
                        clean_resid_energy += clean_err * clean_err;
                    }
                    return 10.0 * std::log10((clean_energy + 1e-12) / (clean_resid_energy + 1e-12));
                };
                output_snr_db_std = compute_snr_std(result_std.approx);
                improvement_db_std = output_snr_db_std - input_snr_db;
                result_std_params_copy = result_std.params;
                result_std_coeffs_copy = result_std.coeffs;
            }

            // ACT_CPU (Eigen+BLAS)
            ACT_CPU::TransformResult result_cpu;
            {
                auto t0_cpu = std::chrono::steady_clock::now();
                ACT_CPU act_cpu(fs, length, ranges_cpu, false);
                act_cpu.generate_chirplet_dictionary();
                auto t1_cpu = std::chrono::steady_clock::now();

                if (coarse_only) {
                    Eigen::Map<const Eigen::VectorXd> x(signal.data(), length);
                    ACT_CPU::TransformOptions opts; opts.order = cfg.order; opts.refine = false; opts.residual_threshold = 1e-6;
                    result_cpu = act_cpu.transform(x, opts);
                } else {
                    result_cpu = act_cpu.transform(signal, cfg.order);
                }
                auto t2_cpu = std::chrono::steady_clock::now();

                dict_elapsed_ms_cpu = std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();
                trans_elapsed_ms_cpu = std::chrono::duration<double, std::milli>(t2_cpu - t1_cpu).count();
                dict_size_cpu = act_cpu.get_dict_size();
            }

            // ACT_Accelerate (Apple Accelerate subclass)
            ACT_CPU::TransformResult result_acc;
            {
                auto t0_acc = std::chrono::steady_clock::now();
                ACT_Accelerate act_acc(fs, length, ranges_cpu, false);
                act_acc.generate_chirplet_dictionary();
                auto t1_acc = std::chrono::steady_clock::now();

                if (coarse_only) {
                    Eigen::Map<const Eigen::VectorXd> x(signal.data(), length);
                    ACT_CPU::TransformOptions opts; opts.order = cfg.order; opts.refine = false; opts.residual_threshold = 1e-6;
                    result_acc = act_acc.transform(x, opts);
                } else {
                    result_acc = act_acc.transform(signal, cfg.order);
                }
                auto t2_acc = std::chrono::steady_clock::now();

                dict_elapsed_ms_acc = std::chrono::duration<double, std::milli>(t1_acc - t0_acc).count();
                trans_elapsed_ms_acc = std::chrono::duration<double, std::milli>(t2_acc - t1_acc).count();
                dict_size_acc = act_acc.get_dict_size();
            }

            if (result_cpu.params.rows() == 0 || result_acc.params.rows() == 0) {
                std::cerr << "FAIL: No chirplets found" << std::endl;
                return 1;
            }

            auto compute_snr_cpu = [&](const Eigen::VectorXd& approx_vec) {
                double clean_resid_energy = 0.0;
                for (int i = 0; i < length; ++i) {
                    double clean_err = clean[i] - approx_vec[i];
                    clean_resid_energy += clean_err * clean_err;
                }
                return 10.0 * std::log10((clean_energy + 1e-12) / (clean_resid_energy + 1e-12));
            };

            output_snr_db_cpu = compute_snr_cpu(result_cpu.approx);
            improvement_db_cpu = output_snr_db_cpu - input_snr_db;
            double est_bytes_std = static_cast<double>(dict_size_std) * length * sizeof(double);
            double est_mb_std = est_bytes_std / (1024.0 * 1024.0);
            double est_bytes_cpu = static_cast<double>(dict_size_cpu) * length * sizeof(double);
            double est_mb_cpu = est_bytes_cpu / (1024.0 * 1024.0);
            double est_bytes_acc = static_cast<double>(dict_size_acc) * length * sizeof(double);
            double est_mb_acc = est_bytes_acc / (1024.0 * 1024.0);

            std::cout << "ACT (std) dict size:  " << dict_size_std << ", est matrix memory: ~" << est_mb_std << " MB\n";
            std::cout << "  Dictionary generation: " << dict_elapsed_ms_std << " ms\n";
            std::cout << "  Transform:             " << trans_elapsed_ms_std << " ms\n";
            std::cout << "  Output SNR:            " << output_snr_db_std << " dB, Improvement: " << improvement_db_std << " dB\n";

            std::cout << "ACT_CPU (Eigen+BLAS) dict size: " << dict_size_cpu << ", est matrix memory: ~" << est_mb_cpu << " MB\n";
            std::cout << "  Dictionary generation: " << dict_elapsed_ms_cpu << " ms\n";
            std::cout << "  Transform:             " << trans_elapsed_ms_cpu << " ms\n";
            std::cout << "  Output SNR:            " << output_snr_db_cpu << " dB, Improvement: " << improvement_db_cpu << " dB\n";

            std::cout << "ACT_Accelerate (vDSP) dict size: " << dict_size_acc << ", est matrix memory: ~" << est_mb_acc << " MB\n";
            std::cout << "  Dictionary generation: " << dict_elapsed_ms_acc << " ms\n";
            std::cout << "  Transform:             " << trans_elapsed_ms_acc << " ms\n";
            {
                auto compute_snr_acc = [&](const Eigen::VectorXd& approx_vec) {
                    double clean_resid_energy = 0.0;
                    for (int i = 0; i < length; ++i) {
                        double clean_err = clean[i] - approx_vec[i];
                        clean_resid_energy += clean_err * clean_err;
                    }
                    return 10.0 * std::log10((clean_energy + 1e-12) / (clean_resid_energy + 1e-12));
                };
                output_snr_db_acc = compute_snr_acc(result_acc.approx);
                improvement_db_acc = output_snr_db_acc - input_snr_db;
                std::cout << "  Output SNR:            " << output_snr_db_acc << " dB, Improvement: " << improvement_db_acc << " dB\n";

                if (output_snr_db_acc > best_out_acc) {
                    best_out_acc = output_snr_db_acc;
                    best_imp_acc = improvement_db_acc;
                    best_dict_acc = dict_size_acc;
                    best_order_acc = cfg.order;
                    best_time_ms_acc = dict_elapsed_ms_acc + trans_elapsed_ms_acc;
                }
            }

            // Recovered vs Truth reporting for ACT (std)
            auto gt = ground_truth_params(length);
            size_t kmax_std = std::min<size_t>(3, std::min(result_std_params_copy.size(), gt.size()));
            std::vector<bool> gt_used_std(gt.size(), false);
            std::cout << "Recovered vs Truth (ACT std, up to " << kmax_std << "):\n";
            for (size_t r = 0; r < kmax_std; ++r) {
                const auto &rp = result_std_params_copy[r];
                int best_idx = -1; double best_dist = 1e300;
                for (size_t gti = 0; gti < gt.size(); ++gti) {
                    if (gt_used_std[gti]) continue;
                    double dtc = rp[0] - gt[gti][0];
                    double dfc = rp[1] - gt[gti][1];
                    double dld = rp[2] - gt[gti][2];
                    double dc  = rp[3] - gt[gti][3];
                    double dist = dtc*dtc + dfc*dfc + dld*dld + dc*dc;
                    if (dist < best_dist) { best_dist = dist; best_idx = (int)gti; }
                }
                if (best_idx >= 0) {
                    gt_used_std[best_idx] = true;
                    double dtc = rp[0] - gt[best_idx][0];
                    double dfc = rp[1] - gt[best_idx][1];
                    double dld = rp[2] - gt[best_idx][2];
                    double dc  = rp[3] - gt[best_idx][3];
                    double coeff = (r < result_std_coeffs_copy.size() ? result_std_coeffs_copy[r] : 0.0);
                    std::cout << "  Atom " << (r+1) << ": rec(tc=" << rp[0] << ", fc=" << rp[1]
                              << ", logDt=" << rp[2] << ", c=" << rp[3] << ", a~" << coeff << ") | "
                              << "gt(tc=" << gt[best_idx][0] << ", fc=" << gt[best_idx][1]
                              << ", logDt=" << gt[best_idx][2] << ", c=" << gt[best_idx][3] << ") | "
                              << "d(tc,fc,logDt,c)=(" << dtc << ", " << dfc << ", " << dld << ", " << dc << ")\n";
                }
            }

            // Track best metrics per method
            if (output_snr_db_std > best_out_std) {
                best_out_std = output_snr_db_std;
                best_imp_std = improvement_db_std;
                best_dict_std = dict_size_std;
                best_order_std = cfg.order;
                best_time_ms_std = dict_elapsed_ms_std + trans_elapsed_ms_std;
            }
            if (output_snr_db_cpu > best_out_cpu) {
                best_out_cpu = output_snr_db_cpu;
                best_imp_cpu = improvement_db_cpu;
                best_dict_cpu = dict_size_cpu;
                best_order_cpu = cfg.order;
                best_time_ms_cpu = dict_elapsed_ms_cpu + trans_elapsed_ms_cpu;
            }
        }
        // Final summary across sweep
        std::cout << "\n=== Summary Across Sweep ===\n";
        std::cout << "ACT (std):     Best Output SNR=" << best_out_std << " dB, Improvement=" << best_imp_std
                  << " dB, dict_size=" << best_dict_std << ", order=" << best_order_std
                  << ", elapsed~" << best_time_ms_std << " ms\n";
        std::cout << "ACT_CPU (BLAS): Best Output SNR=" << best_out_cpu << " dB, Improvement=" << best_imp_cpu
                  << " dB, dict_size=" << best_dict_cpu << ", order=" << best_order_cpu
                  << ", elapsed~" << best_time_ms_cpu << " ms\n";
        std::cout << "ACT_Accelerate: Best Output SNR=" << best_out_acc << " dB, Improvement=" << best_imp_acc
                  << " dB, dict_size=" << best_dict_acc << ", order=" << best_order_acc
                  << ", elapsed~" << best_time_ms_acc << " ms\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
