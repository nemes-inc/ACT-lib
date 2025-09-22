#include "ACT.h"
#include "ACT_CPU.h"
#include "ACT_Accelerate.h"
#include "ACT_MLX.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstdlib>

// MLX backend inherits ACT_Accelerate CPU path; GPU acceleration can be wired later

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
    
    double elapsed_s() {
        return elapsed_ms() / 1000.0; // Convert to seconds
    }
};

// Generate synthetic EEG-like signal
std::vector<double> generate_eeg_signal(int length, double fs, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 0.1);
    
    std::vector<double> signal(length, 0.0);
    
    // Add multiple frequency components typical of EEG
    for (int i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / fs;
        
        // Alpha waves (8-12 Hz)
        signal[i] += 0.5 * std::sin(2 * M_PI * 10.0 * t);
        
        // Beta waves (13-30 Hz) 
        signal[i] += 0.3 * std::sin(2 * M_PI * 20.0 * t);
        
        // Theta waves (4-8 Hz)
        signal[i] += 0.4 * std::sin(2 * M_PI * 6.0 * t);
        
        // Add some chirp-like components
        double chirp_rate = 2.0; // Hz/s
        signal[i] += 0.2 * std::sin(2 * M_PI * (8.0 * t + 0.5 * chirp_rate * t * t));
        
        // Add noise
        signal[i] += noise(gen);
    }
    
    return signal;
}

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void print_timing(const std::string& operation, double time_ms, const std::string& details = "") {
    std::cout << std::left << std::setw(30) << operation 
              << std::right << std::setw(10) << std::fixed << std::setprecision(2) 
              << time_ms << " ms";
    if (!details.empty()) {
        std::cout << "  (" << details << ")";
    }
    std::cout << std::endl;
}

int main() {
    print_separator("C++ ACT PROFILING TEST - EEG SCALE");
    
    // EEG-typical parameters
    const double FS = 256.0;           // 256 Hz sampling rate (typical for EEG)
    const int SIGNAL_LENGTH = 512;   // 2 seconds of data
    const int TRANSFORM_ORDER = 10;   // Higher order for detailed analysis
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Sampling Rate: " << FS << " Hz" << std::endl;
    std::cout << "  Signal Length: " << SIGNAL_LENGTH << " samples (" 
              << SIGNAL_LENGTH/FS << " seconds)" << std::endl;
    std::cout << "  Transform Order: " << TRANSFORM_ORDER << " chirplets" << std::endl;
    
    // Define EEG-appropriate parameter ranges
    ACT::ParameterRanges eeg_ranges(
        0, SIGNAL_LENGTH-1, 16,    // tc: time center (64 values)
        0.5, 20.0, 1,           // fc: 0.5-50 Hz (100 values, covers EEG spectrum)
        -1.9, -0.10, 0.08,          // logDt: duration range (16 values)
        -20.0, 20.0, 1.0          // c: chirp rate (21 values)
    );
    
    // Calculate expected dictionary size
    int tc_count = static_cast<int>((SIGNAL_LENGTH-1 - 0) / 16) + 1;
    int fc_count = static_cast<int>((50.0 - 0.5) / 0.5) + 1;
    int logDt_count = static_cast<int>((-1.0 - (-4.0)) / 0.2) + 1;
    int c_count = static_cast<int>((20.0 - (-20.0)) / 2.0) + 1;
    int expected_dict_size = tc_count * fc_count * logDt_count * c_count;
    
    std::cout << "  Expected Dictionary Size: " << expected_dict_size << " chirplets" << std::endl;
    
    Timer timer;
    
    print_separator("DICTIONARY GENERATION");
    
    // Backend selection via env var
    std::string backend = "cpu"; // default
    if (const char* e = std::getenv("ACT_PROFILE_BACKEND")) backend = std::string(e);
    std::string prec = "double"; // default
    if (const char* p = std::getenv("ACT_PROFILE_PREC")) prec = std::string(p);
    bool use_float = (prec == "float" || prec == "float32" || prec == "f32");
    bool coarse_only = false;
    if (const char* e = std::getenv("ACT_COARSE_ONLY")) {
        if (std::string(e) == "1" || std::string(e) == "true" || std::string(e) == "TRUE") coarse_only = true;
    }
    std::cout << "Backend: " << backend << std::endl;
    std::cout << "Precision: " << (use_float ? "float32" : "double") << std::endl;
    std::cout << "Coarse Only: " << coarse_only << std::endl;

    // Print memory estimate after precision is known
    size_t bytes_per_elem = use_float ? sizeof(float) : sizeof(double);
    std::cout << "  Memory Estimate: ~" << (expected_dict_size * SIGNAL_LENGTH * bytes_per_elem) / (1024*1024) 
              << " MB" << std::endl;

    // Mirror ranges for ACT_CPU-based classes
    ACT_CPU::ParameterRanges cpu_ranges(
        eeg_ranges.tc_min, eeg_ranges.tc_max, eeg_ranges.tc_step,
        eeg_ranges.fc_min, eeg_ranges.fc_max, eeg_ranges.fc_step,
        eeg_ranges.logDt_min, eeg_ranges.logDt_max, eeg_ranges.logDt_step,
        eeg_ranges.c_min, eeg_ranges.c_max, eeg_ranges.c_step
    );

    // Initialize selected backend and generate dictionary
    timer.start();
    int actual_dict_size = 0;
    std::unique_ptr<ACT> act_legacy;
    std::unique_ptr<ACT_CPU> act_cpu;
    std::unique_ptr<ACT_Accelerate> act_accel;
    std::unique_ptr<ACT_MLX> act_mlx;
    // Float32 variants
    std::unique_ptr<ACT_CPU_f> act_cpu_f;
    std::unique_ptr<ACT_Accelerate_f> act_accel_f;
    std::unique_ptr<ACT_MLX_f> act_mlx_f;

    if (backend == "act" || backend == "legacy") {
        act_legacy.reset(new ACT(FS, SIGNAL_LENGTH, eeg_ranges, false, true));
        double init_time = timer.elapsed_ms();
        print_timing("ACT Initialization", init_time);
        timer.start();
        actual_dict_size = act_legacy->generate_chirplet_dictionary();
    } else if (backend == "accel") {
        if (use_float) {
            ACT_CPU_f::ParameterRanges cpu_ranges_f(
                cpu_ranges.tc_min, cpu_ranges.tc_max, cpu_ranges.tc_step,
                cpu_ranges.fc_min, cpu_ranges.fc_max, cpu_ranges.fc_step,
                cpu_ranges.logDt_min, cpu_ranges.logDt_max, cpu_ranges.logDt_step,
                cpu_ranges.c_min, cpu_ranges.c_max, cpu_ranges.c_step
            );
            act_accel_f.reset(new ACT_Accelerate_f(FS, SIGNAL_LENGTH, cpu_ranges_f, false));
        } else {
            act_accel.reset(new ACT_Accelerate(FS, SIGNAL_LENGTH, cpu_ranges, false));
        }
        double init_time = timer.elapsed_ms();
        print_timing("ACCEL Initialization", init_time);
        timer.start();
        actual_dict_size = (use_float ? act_accel_f->generate_chirplet_dictionary() : act_accel->generate_chirplet_dictionary());
    } else if (backend == "mlx") {
        if (use_float) {
            ACT_CPU_f::ParameterRanges cpu_ranges_f(
                cpu_ranges.tc_min, cpu_ranges.tc_max, cpu_ranges.tc_step,
                cpu_ranges.fc_min, cpu_ranges.fc_max, cpu_ranges.fc_step,
                cpu_ranges.logDt_min, cpu_ranges.logDt_max, cpu_ranges.logDt_step,
                cpu_ranges.c_min, cpu_ranges.c_max, cpu_ranges.c_step
            );
            act_mlx_f.reset(new ACT_MLX_f(FS, SIGNAL_LENGTH, cpu_ranges_f, true));
        } else {
            act_mlx.reset(new ACT_MLX(FS, SIGNAL_LENGTH, cpu_ranges, true));
        }
        double init_time = timer.elapsed_ms();
        print_timing("MLX Initialization", init_time);
        timer.start();
        actual_dict_size = (use_float ? act_mlx_f->generate_chirplet_dictionary() : act_mlx->generate_chirplet_dictionary());
    } else { // default CPU
        if (use_float) {
            ACT_CPU_f::ParameterRanges cpu_ranges_f(
                cpu_ranges.tc_min, cpu_ranges.tc_max, cpu_ranges.tc_step,
                cpu_ranges.fc_min, cpu_ranges.fc_max, cpu_ranges.fc_step,
                cpu_ranges.logDt_min, cpu_ranges.logDt_max, cpu_ranges.logDt_step,
                cpu_ranges.c_min, cpu_ranges.c_max, cpu_ranges.c_step
            );
            act_cpu_f.reset(new ACT_CPU_f(FS, SIGNAL_LENGTH, cpu_ranges_f, false));
        } else {
            act_cpu.reset(new ACT_CPU(FS, SIGNAL_LENGTH, cpu_ranges, false));
        }
        double init_time = timer.elapsed_ms();
        print_timing("CPU Initialization", init_time);
        timer.start();
        actual_dict_size = (use_float ? act_cpu_f->generate_chirplet_dictionary() : act_cpu->generate_chirplet_dictionary());
    }
    double dict_gen_time = timer.elapsed_s();
    print_timing("Dictionary Generation", dict_gen_time * 1000, 
                 std::to_string(actual_dict_size) + " chirplets");
    
    std::cout << "Dictionary generation rate: " 
              << static_cast<int>(actual_dict_size / dict_gen_time) << " chirplets/second" << std::endl;
    
    print_separator("SIGNAL GENERATION AND ANALYSIS");
    
    // Generate multiple test signals
    const int NUM_SIGNALS = 5;
    std::vector<std::vector<double>> test_signals;
    
    timer.start();
    for (int i = 0; i < NUM_SIGNALS; ++i) {
        test_signals.push_back(generate_eeg_signal(SIGNAL_LENGTH, FS, 42 + i));
    }
    double signal_gen_time = timer.elapsed_ms();
    print_timing("Signal Generation", signal_gen_time, 
                 std::to_string(NUM_SIGNALS) + " signals");
    
    // Analyze each signal
    std::vector<double> search_times;
    std::vector<double> transform_times;
    std::vector<double> total_times;
    std::vector<double> snr_values;
    
    for (int i = 0; i < NUM_SIGNALS; ++i) {
        std::cout << "\nAnalyzing Signal " << (i+1) << "/" << NUM_SIGNALS << ":" << std::endl;
        
        // Perform one search to warm up the dictionary in memory
        if (act_legacy) act_legacy->search_dictionary(test_signals[i]);
        else if (act_mlx || act_mlx_f) {
            if (use_float) {
                const auto& s = test_signals[i];
                std::vector<float> sf(s.size());
                for (size_t j=0;j<s.size();++j) sf[j] = static_cast<float>(s[j]);
                act_mlx_f->search_dictionary(sf);
            } else {
                act_mlx->search_dictionary(test_signals[i]);
            }
        }
        else if (act_accel || act_accel_f) {
            if (use_float) {
                const auto& s = test_signals[i];
                std::vector<float> sf(s.size());
                for (size_t j=0;j<s.size();++j) sf[j] = static_cast<float>(s[j]);
                act_accel_f->search_dictionary(sf);
            } else {
                act_accel->search_dictionary(test_signals[i]);
            }
        }
        else if (act_cpu || act_cpu_f) {
            if (use_float) {
                const auto& s = test_signals[i];
                std::vector<float> sf(s.size());
                for (size_t j=0;j<s.size();++j) sf[j] = static_cast<float>(s[j]);
                act_cpu_f->search_dictionary(sf);
            } else {
                act_cpu->search_dictionary(test_signals[i]);
            }
        }
        
        // Dictionary search timing
        timer.start();
        std::pair<int,double> search_result;
        if (act_legacy) search_result = act_legacy->search_dictionary(test_signals[i]);
        else if (act_mlx || act_mlx_f) {
            if (use_float) {
                const auto& s = test_signals[i];
                std::vector<float> sf(s.size());
                for (size_t j=0;j<s.size();++j) sf[j] = static_cast<float>(s[j]);
                auto r = act_mlx_f->search_dictionary(sf);
                search_result = {r.first, static_cast<double>(r.second)};
            } else {
                search_result = act_mlx->search_dictionary(test_signals[i]);
            }
        }
        else if (act_accel || act_accel_f) {
            if (use_float) {
                const auto& s = test_signals[i];
                std::vector<float> sf(s.size());
                for (size_t j=0;j<s.size();++j) sf[j] = static_cast<float>(s[j]);
                auto r = act_accel_f->search_dictionary(sf);
                search_result = {r.first, static_cast<double>(r.second)};
            } else {
                search_result = act_accel->search_dictionary(test_signals[i]);
            }
        }
        else if (act_cpu || act_cpu_f) {
            if (use_float) {
                const auto& s = test_signals[i];
                std::vector<float> sf(s.size());
                for (size_t j=0;j<s.size();++j) sf[j] = static_cast<float>(s[j]);
                auto r = act_cpu_f->search_dictionary(sf);
                search_result = {r.first, static_cast<double>(r.second)};
            } else {
                search_result = act_cpu->search_dictionary(test_signals[i]);
            }
        }
        double search_time = timer.elapsed_ms();
        search_times.push_back(search_time);
        print_timing("  Dictionary Search", search_time, 
                     "best match: " + std::to_string(search_result.first));
        
        // Full transform timing
        timer.start();
        // Transform
        ACT::TransformResult result;
        if (act_legacy) {
            auto r = act_legacy->transform(test_signals[i], TRANSFORM_ORDER, false);
            result = r;
        } else if (act_cpu || act_accel || act_mlx || act_cpu_f || act_accel_f || act_mlx_f) {
            if (coarse_only) {
                std::cout << "  Coarse Only Transform" << std::endl;
                const auto& s = test_signals[i];
                ACT::TransformResult result_tmp;
                if (use_float) {
                    ACT_CPU_f::TransformOptions opts; opts.order = TRANSFORM_ORDER; opts.refine = false; opts.residual_threshold = 1e-6;
                    std::vector<float> sf(s.size()); for (size_t j=0;j<s.size();++j) sf[j] = static_cast<float>(s[j]);
                    Eigen::Map<const act::VecX<float>> x(sf.data(), (int)sf.size());
                    ACT_CPU_f::TransformResult r;
                    if (act_accel_f) r = act_accel_f->transform(x, opts);
                    else if (act_mlx_f) r = act_mlx_f->transform(x, opts);
                    else r = act_cpu_f->transform(x, opts);
                    // Convert r to ACT::TransformResult below
                    result.params.resize(r.params.rows());
                    for (int rr = 0; rr < r.params.rows(); ++rr) {
                        result.params[rr] = { r.params(rr,0), r.params(rr,1), r.params(rr,2), r.params(rr,3) };
                    }
                    result.coeffs.resize(r.coeffs.size());
                    for (int k = 0; k < r.coeffs.size(); ++k) result.coeffs[k] = r.coeffs[k];
                    result.signal.resize(r.signal.size());
                    for (int k = 0; k < r.signal.size(); ++k) result.signal[k] = r.signal[k];
                    result.residue.resize(r.residue.size());
                    for (int k = 0; k < r.residue.size(); ++k) result.residue[k] = r.residue[k];
                    result.approx.resize(r.approx.size());
                    for (int k = 0; k < r.approx.size(); ++k) result.approx[k] = r.approx[k];
                    result.error = r.error;
                } else {
                    ACT_CPU::TransformOptions opts; opts.order = TRANSFORM_ORDER; opts.refine = false; opts.residual_threshold = 1e-6;
                    Eigen::Map<const Eigen::VectorXd> x(s.data(), (int)s.size());
                    ACT_CPU::TransformResult r;
                    if (act_accel) r = act_accel->transform(x, opts);
                    else if (act_mlx) r = act_mlx->transform(x, opts);
                    else r = act_cpu->transform(x, opts);
                    result.params.resize(r.params.rows());
                    for (int rr = 0; rr < r.params.rows(); ++rr) {
                        result.params[rr] = { r.params(rr,0), r.params(rr,1), r.params(rr,2), r.params(rr,3) };
                    }
                    result.coeffs.resize(r.coeffs.size());
                    for (int k = 0; k < r.coeffs.size(); ++k) result.coeffs[k] = r.coeffs[k];
                    result.signal.resize(r.signal.size());
                    for (int k = 0; k < r.signal.size(); ++k) result.signal[k] = r.signal[k];
                    result.residue.resize(r.residue.size());
                    for (int k = 0; k < r.residue.size(); ++k) result.residue[k] = r.residue[k];
                    result.approx.resize(r.approx.size());
                    for (int k = 0; k < r.approx.size(); ++k) result.approx[k] = r.approx[k];
                    result.error = r.error;
                }
            } else {
                std::cout << "  Full Transform with BFGS" << std::endl;
                // Use vector overload
                ACT::TransformResult result_tmp;
                if (use_float) {
                    std::vector<float> sf(test_signals[i].size());
                    for (size_t j=0;j<sf.size();++j) sf[j] = static_cast<float>(test_signals[i][j]);
                    ACT_CPU_f::TransformResult r = (act_accel_f ? act_accel_f->transform(sf, TRANSFORM_ORDER, 1e-6f)
                                                                : (act_mlx_f ? act_mlx_f->transform(sf, TRANSFORM_ORDER, 1e-6f)
                                                                              : act_cpu_f->transform(sf, TRANSFORM_ORDER, 1e-6f)));
                    result.params.resize(r.params.rows());
                    for (int rr = 0; rr < r.params.rows(); ++rr) result.params[rr] = { r.params(rr,0), r.params(rr,1), r.params(rr,2), r.params(rr,3) };
                    result.coeffs.resize(r.coeffs.size());
                    for (int k = 0; k < r.coeffs.size(); ++k) result.coeffs[k] = r.coeffs[k];
                    result.signal.resize(r.signal.size());
                    for (int k = 0; k < r.signal.size(); ++k) result.signal[k] = r.signal[k];
                    result.residue.resize(r.residue.size());
                    for (int k = 0; k < r.residue.size(); ++k) result.residue[k] = r.residue[k];
                    result.approx.resize(r.approx.size());
                    for (int k = 0; k < r.approx.size(); ++k) result.approx[k] = r.approx[k];
                    result.error = r.error;
                } else {
                    ACT_CPU::TransformResult r = (act_accel ? act_accel->transform(test_signals[i], TRANSFORM_ORDER, 1e-6)
                                                            : (act_mlx ? act_mlx->transform(test_signals[i], TRANSFORM_ORDER, 1e-6)
                                                                       : act_cpu->transform(test_signals[i], TRANSFORM_ORDER, 1e-6)));
                    result.params.resize(r.params.rows());
                    for (int rr = 0; rr < r.params.rows(); ++rr) result.params[rr] = { r.params(rr,0), r.params(rr,1), r.params(rr,2), r.params(rr,3) };
                    result.coeffs.resize(r.coeffs.size());
                    for (int k = 0; k < r.coeffs.size(); ++k) result.coeffs[k] = r.coeffs[k];
                    result.signal.resize(r.signal.size());
                    for (int k = 0; k < r.signal.size(); ++k) result.signal[k] = r.signal[k];
                    result.residue.resize(r.residue.size());
                    for (int k = 0; k < r.residue.size(); ++k) result.residue[k] = r.residue[k];
                    result.approx.resize(r.approx.size());
                    for (int k = 0; k < r.approx.size(); ++k) result.approx[k] = r.approx[k];
                    result.error = r.error;
                }
            }
        }
        double transform_time = timer.elapsed_s();
        transform_times.push_back(transform_time * 1000);
        total_times.push_back(search_time + transform_time * 1000);
        
        print_timing("  Full Transform", transform_time * 1000, 
                     std::to_string(TRANSFORM_ORDER) + " chirplets");
        
        // Calculate SNR (guard against size mismatches)
        size_t n_signal = test_signals[i].size();
        size_t n_residue = result.residue.size();
        size_t n = std::min(n_signal, n_residue);
        double signal_energy = 0.0, residue_energy = 0.0;
        if (n == 0) {
            std::cerr << "Warning: empty or mismatched residue (signal=" << n_signal
                      << ", residue=" << n_residue << ") — skipping SNR for this signal." << std::endl;
        } else {
            for (size_t j = 0; j < n; ++j) {
                signal_energy += test_signals[i][j] * test_signals[i][j];
                residue_energy += result.residue[j] * result.residue[j];
            }
        }
        double snr = (residue_energy > 0.0 ? 10.0 * std::log10(signal_energy / residue_energy) : 0.0);
        snr_values.push_back(snr);
        
        std::cout << "  SNR: " << std::fixed << std::setprecision(2) << snr << " dB" << std::endl;
        std::cout << "  Residual Error: " << result.error << std::endl;
    }
    
    print_separator("PERFORMANCE STATISTICS");
    
    // Calculate statistics
    auto calc_stats = [](const std::vector<double>& values) {
        double sum = 0.0, min_val = values[0], max_val = values[0];
        for (double val : values) {
            sum += val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        double mean = sum / values.size();
        return std::make_tuple(mean, min_val, max_val);
    };
    
    auto [mean_search, min_search, max_search] = calc_stats(search_times);
    auto [mean_transform, min_transform, max_transform] = calc_stats(transform_times);
    auto [mean_total, min_total, max_total] = calc_stats(total_times);
    auto [mean_snr, min_snr, max_snr] = calc_stats(snr_values);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Dictionary Search Times:" << std::endl;
    std::cout << "  Mean: " << mean_search << " ms, Range: [" << min_search << ", " << max_search << "] ms" << std::endl;
    
    std::cout << "Transform Times:" << std::endl;
    std::cout << "  Mean: " << mean_transform << " ms, Range: [" << min_transform << ", " << max_transform << "] ms" << std::endl;
    
    std::cout << "Total Analysis Times:" << std::endl;
    std::cout << "  Mean: " << mean_total << " ms, Range: [" << min_total << ", " << max_total << "] ms" << std::endl;
    
    std::cout << "Signal-to-Noise Ratios:" << std::endl;
    std::cout << "  Mean: " << mean_snr << " dB, Range: [" << min_snr << ", " << max_snr << "] dB" << std::endl;
    
    print_separator("THROUGHPUT ANALYSIS");
    
    // Calculate throughput metrics
    double avg_analysis_time_s = mean_total / 1000.0;
    double signal_duration_s = SIGNAL_LENGTH / FS;
    double realtime_factor = signal_duration_s / avg_analysis_time_s;
    
    std::cout << "Signal Duration: " << signal_duration_s << " seconds" << std::endl;
    std::cout << "Average Analysis Time: " << avg_analysis_time_s << " seconds" << std::endl;
    std::cout << "Real-time Factor: " << std::fixed << std::setprecision(1) << realtime_factor << "x" << std::endl;
    
    if (realtime_factor >= 1.0) {
        std::cout << "✓ Real-time processing capable!" << std::endl;
    } else {
        std::cout << "⚠ Not real-time (would need " << std::setprecision(1) 
                  << (1.0/realtime_factor) << "x speedup)" << std::endl;
    }
    
    // Memory usage estimate
    size_t dict_bytes_per_elem = (std::getenv("ACT_PROFILE_PREC") &&
                                  (std::string(std::getenv("ACT_PROFILE_PREC")) == "float" ||
                                   std::string(std::getenv("ACT_PROFILE_PREC")) == "float32" ||
                                   std::string(std::getenv("ACT_PROFILE_PREC")) == "f32")) ? sizeof(float) : sizeof(double);
    double dict_memory_mb = (actual_dict_size * SIGNAL_LENGTH * dict_bytes_per_elem) / (1024.0 * 1024.0);
    std::cout << "Dictionary Memory Usage: " << std::setprecision(1) << dict_memory_mb << " MB" << std::endl;
    
    print_separator("OPTIMIZATION ANALYSIS");
    
    // Note: BFGS optimization is called internally during transform
    // Optimization overhead is included in transform timing above
    std::cout << "BFGS optimization is integrated into the transform process." << std::endl;
    std::cout << "Optimization timing is included in the transform measurements." << std::endl;
    
    // Estimate optimization overhead per chirplet
    double opt_overhead_per_chirplet = (mean_transform - mean_search) / TRANSFORM_ORDER;
    std::cout << "Optimization overhead per chirplet: " << std::setprecision(2) 
              << opt_overhead_per_chirplet << " ms" << std::endl;
    
    print_separator("SUMMARY");
    
    std::cout << "Dictionary: " << actual_dict_size << " chirplets, " 
              << std::setprecision(1) << dict_memory_mb << " MB" << std::endl;
    std::cout << "Average Analysis: " << std::setprecision(2) << mean_total << " ms (" 
              << std::setprecision(1) << realtime_factor << "x real-time)" << std::endl;
    std::cout << "Average SNR: " << std::setprecision(2) << mean_snr << " dB" << std::endl;
    std::cout << "Bounded BFGS: Integrated in transform (" 
              << std::setprecision(1) << opt_overhead_per_chirplet << " ms/chirplet)" << std::endl;
    
    std::cout << "\n=== PROFILING COMPLETE ===" << std::endl;
    
    return 0;
}
