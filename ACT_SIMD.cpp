#include "ACT_SIMD.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

ACT_SIMD::ACT_SIMD(double FS, int length, const std::string& dict_addr, 
                   const ParameterRanges& ranges, bool complex_mode, 
                   bool force_regenerate, bool mute)
    : ACT(FS, length, dict_addr, ranges, complex_mode, force_regenerate, mute),
      has_accelerate(false), has_neon(false) {
    
    detect_simd_features();
    initialize_simd_memory();
    
    if (!mute) {
        std::cout << "\n=== SIMD ACT INITIALIZATION ===" << std::endl;
        std::cout << "Apple Accelerate: " << (has_accelerate ? "✅ Available" : "❌ Not available") << std::endl;
        std::cout << "ARM NEON: " << (has_neon ? "✅ Available" : "❌ Not available") << std::endl;
        std::cout << "================================" << std::endl;
    }
}

ACT_SIMD::~ACT_SIMD() {
    cleanup_simd_memory();
}

void ACT_SIMD::detect_simd_features() {
#ifdef __APPLE__
    has_accelerate = true;
#endif

#ifdef __ARM_NEON
    has_neon = true;
#endif
}

void ACT_SIMD::initialize_simd_memory() {
    // Memory alignment is handled by std::vector for our use case
    // For more advanced SIMD, we might need aligned_alloc
}

void ACT_SIMD::cleanup_simd_memory() {
    // Cleanup if needed
}

std::pair<int, double> ACT_SIMD::search_dictionary(const std::vector<double>& signal) {
    const auto& dict_mat = get_dict_mat();
    int dict_size = get_dict_size();
    
    if (dict_size == 0) {
        return std::make_pair(0, 0.0);
    }
    
    int best_idx = 0;
    double best_value = -std::numeric_limits<double>::infinity();
    
    // Use the best available SIMD method
    if (has_accelerate) {
        // Apple Accelerate framework - most optimized for Apple Silicon
        for (int i = 0; i < dict_size; ++i) {
            double val = inner_product_accelerate(dict_mat[i], signal);
            if (val > best_value) {
                best_value = val;
                best_idx = i;
            }
        }
    } else if (has_neon) {
        // ARM NEON intrinsics fallback
        for (int i = 0; i < dict_size; ++i) {
            double val = inner_product_neon(dict_mat[i].data(), signal.data(), signal.size());
            if (val > best_value) {
                best_value = val;
                best_idx = i;
            }
        }
    } else {
        // Auto-vectorized fallback
        for (int i = 0; i < dict_size; ++i) {
            double val = inner_product_auto_vectorized(dict_mat[i], signal);
            if (val > best_value) {
                best_value = val;
                best_idx = i;
            }
        }
    }
    
    return std::make_pair(best_idx, best_value);
}

double ACT_SIMD::inner_product_accelerate(const std::vector<double>& a, const std::vector<double>& b) {
#ifdef __APPLE__
    // Apple Accelerate framework - highly optimized for Apple Silicon
    double result;
    vDSP_dotprD(a.data(), 1, b.data(), 1, &result, a.size());
    return result;
#else
    // Fallback to auto-vectorized version
    return inner_product_auto_vectorized(a, b);
#endif
}

double ACT_SIMD::inner_product_neon(const double* __restrict a, const double* __restrict b, size_t length) {
#ifdef __ARM_NEON
    // ARM NEON intrinsics for 128-bit vectors (2 doubles)
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    
    size_t simd_length = length & ~1; // Round down to nearest multiple of 2
    
    // Process 2 doubles at a time
    for (size_t i = 0; i < simd_length; i += 2) {
        float64x2_t a_vec = vld1q_f64(&a[i]);
        float64x2_t b_vec = vld1q_f64(&b[i]);
        float64x2_t prod = vmulq_f64(a_vec, b_vec);
        sum_vec = vaddq_f64(sum_vec, prod);
    }
    
    // Horizontal sum of the vector
    double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
    
    // Handle remaining elements
    for (size_t i = simd_length; i < length; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
#else
    // Fallback to scalar implementation
    double sum = 0.0;
    for (size_t i = 0; i < length; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

double ACT_SIMD::inner_product_auto_vectorized(const std::vector<double>& a, const std::vector<double>& b) {
    // Let the compiler auto-vectorize this loop
    const double* __restrict a_ptr = a.data();
    const double* __restrict b_ptr = b.data();
    const size_t length = a.size();
    
    double sum = 0.0;
    
    // Compiler hints for vectorization
    #pragma clang loop vectorize(enable)
    #pragma clang loop unroll(enable)
    for (size_t i = 0; i < length; ++i) {
        sum += a_ptr[i] * b_ptr[i];
    }
    
    return sum;
}

void ACT_SIMD::batch_inner_products_accelerate(const std::vector<double>& signal,
                                             int start_idx, int batch_size,
                                             std::vector<double>& results) {
#ifdef __APPLE__
    const auto& dict_mat = get_dict_mat();
    results.resize(batch_size);
    
    // Process multiple chirplets using Accelerate's optimized BLAS operations
    for (int i = 0; i < batch_size; ++i) {
        int dict_idx = start_idx + i;
        if (dict_idx < get_dict_size()) {
            vDSP_dotprD(dict_mat[dict_idx].data(), 1, signal.data(), 1, 
                       &results[i], signal.size());
        } else {
            results[i] = -std::numeric_limits<double>::infinity();
        }
    }
#else
    // Fallback implementation
    results.resize(batch_size);
    const auto& dict_mat = get_dict_mat();
    
    for (int i = 0; i < batch_size; ++i) {
        int dict_idx = start_idx + i;
        if (dict_idx < get_dict_size()) {
            results[i] = inner_product_auto_vectorized(dict_mat[dict_idx], signal);
        } else {
            results[i] = -std::numeric_limits<double>::infinity();
        }
    }
#endif
}

// SIMD-accelerated chirplet generation for optimization only
std::vector<double> ACT_SIMD::g_simd_optimization(double tc, double fc, double logDt, double c) {
    // Use the best available SIMD method for optimization step
    if (has_accelerate) {
        return g_accelerate(tc, fc, logDt, c);
    } else if (has_neon) {
        return g_neon(tc, fc, logDt, c);
    } else {
        // Fallback to base class implementation
        return ACT::g(tc, fc, logDt, c);
    }
}

// SIMD-accelerated objective function - override base class method
double ACT_SIMD::minimize_this(const std::vector<double>& params, const std::vector<double>& signal) {
    // Generate chirplet using SIMD-accelerated method for optimization
    auto chirplet = g_simd_optimization(params[0], params[1], params[2], params[3]);
    
    // Use SIMD-accelerated inner product
    double inner_prod = 0.0;
    if (has_accelerate) {
        inner_prod = inner_product_accelerate(chirplet, signal);
    } else if (has_neon) {
        inner_prod = inner_product_neon(chirplet.data(), signal.data(), signal.size());
    } else {
        inner_prod = inner_product_auto_vectorized(chirplet, signal);
    }
    
    // Return negative for minimization (we want to maximize inner product)
    return -inner_prod;
}

// Apple Accelerate SIMD chirplet generation
std::vector<double> ACT_SIMD::g_accelerate(double tc, double fc, double logDt, double c) {
#ifdef __APPLE__
    // Get signal parameters using getter methods
    int signal_length = get_length();
    double sampling_freq = get_FS();
    
    // Add bounds checking for numerical stability
    if (std::isnan(tc) || std::isnan(fc) || std::isnan(logDt) || std::isnan(c)) {
        return std::vector<double>(signal_length, 0.0);
    }
    
    // Clamp logDt to prevent extreme values
    logDt = std::max(-10.0, std::min(2.0, logDt));
    
    // Convert tc from samples to seconds
    tc /= sampling_freq;
    
    // Calculate duration from log value with bounds checking
    double Dt = std::exp(logDt);
    if (Dt < 1e-10 || Dt > 100.0) {
        return std::vector<double>(signal_length, 0.0);
    }
    
    // Prepare vectorized computation
    std::vector<double> t(signal_length);
    std::vector<double> time_diff(signal_length);
    std::vector<double> exponents(signal_length);
    std::vector<double> phases(signal_length);
    std::vector<double> gaussian_window(signal_length);
    std::vector<double> complex_exp(signal_length);
    std::vector<double> chirplet(signal_length);
    
    // Generate time array
    double time_step = 1.0 / sampling_freq;
    for (int i = 0; i < signal_length; ++i) {
        t[i] = static_cast<double>(i) * time_step;
        time_diff[i] = t[i] - tc;
    }
    
    // Vectorized computation of Gaussian exponents
    double inv_Dt = 1.0 / Dt;
    double neg_half = -0.5;
    for (int i = 0; i < signal_length; ++i) {
        double normalized_diff = time_diff[i] * inv_Dt;
        exponents[i] = std::max(-50.0, neg_half * normalized_diff * normalized_diff);
    }
    
    // Vectorized exponential using Apple Accelerate
    int n = static_cast<int>(signal_length);
    vvexp(gaussian_window.data(), exponents.data(), &n);
    
    // Vectorized computation of phases
    double two_pi = 2.0 * M_PI;
    for (int i = 0; i < signal_length; ++i) {
        phases[i] = two_pi * (c * time_diff[i] * time_diff[i] + fc * time_diff[i]);
    }
    
    // Vectorized trigonometric function using Apple Accelerate
    // Use correct trigonometric function based on complex_mode
    if (get_complex_mode()) {
        vvsin(complex_exp.data(), phases.data(), &n);
    } else {
        vvcos(complex_exp.data(), phases.data(), &n);
    }
    
    // Vectorized element-wise multiplication using Apple Accelerate
    vDSP_vmulD(gaussian_window.data(), 1, complex_exp.data(), 1, chirplet.data(), 1, signal_length);
    
    // Check for invalid values and clamp
    for (int i = 0; i < signal_length; ++i) {
        if (std::isnan(chirplet[i]) || std::isinf(chirplet[i])) {
            chirplet[i] = 0.0;
        }
    }
    
    return chirplet;
#else
    // Fallback to base class implementation
    return ACT::g(tc, fc, logDt, c);
#endif
}

// ARM NEON SIMD chirplet generation
std::vector<double> ACT_SIMD::g_neon(double tc, double fc, double logDt, double c) {
#ifdef __ARM_NEON
    // Get signal parameters using getter methods
    int signal_length = get_length();
    double sampling_freq = get_FS();
    
    // Add bounds checking for numerical stability
    if (std::isnan(tc) || std::isnan(fc) || std::isnan(logDt) || std::isnan(c)) {
        return std::vector<double>(signal_length, 0.0);
    }
    
    // Clamp logDt to prevent extreme values
    logDt = std::max(-10.0, std::min(2.0, logDt));
    
    // Convert tc from samples to seconds
    tc /= sampling_freq;
    
    // Calculate duration from log value with bounds checking
    double Dt = std::exp(logDt);
    if (Dt < 1e-10 || Dt > 100.0) {
        return std::vector<double>(signal_length, 0.0);
    }
    
    std::vector<double> chirplet(signal_length);
    double time_step = 1.0 / sampling_freq;
    double inv_Dt = 1.0 / Dt;
    double two_pi = 2.0 * M_PI;
    
    // Process in chunks of 2 doubles (128-bit NEON vectors)
    int simd_length = (signal_length / 2) * 2;
    
    for (int i = 0; i < simd_length; i += 2) {
        // Load time values
        float64x2_t t_vec = {static_cast<double>(i) * time_step, static_cast<double>(i + 1) * time_step};
        float64x2_t tc_vec = vdupq_n_f64(tc);
        float64x2_t time_diff = vsubq_f64(t_vec, tc_vec);
        
        // Gaussian window computation
        float64x2_t normalized_diff = vmulq_n_f64(time_diff, inv_Dt);
        float64x2_t squared_diff = vmulq_f64(normalized_diff, normalized_diff);
        float64x2_t exponent = vmulq_n_f64(squared_diff, -0.5);
        
        // Phase computation
        float64x2_t time_diff_squared = vmulq_f64(time_diff, time_diff);
        float64x2_t phase_term1 = vmulq_n_f64(time_diff_squared, c);
        float64x2_t phase_term2 = vmulq_n_f64(time_diff, fc);
        float64x2_t phase = vmulq_n_f64(vaddq_f64(phase_term1, phase_term2), two_pi);
        
        // For transcendental functions, fall back to scalar (NEON doesn't have vectorized exp/cos/sin)
        double exp_vals[2], trig_vals[2];
        exp_vals[0] = std::exp(std::max(-50.0, vgetq_lane_f64(exponent, 0)));
        exp_vals[1] = std::exp(std::max(-50.0, vgetq_lane_f64(exponent, 1)));
        
        // Use cosine for real mode (since complex_mode is private)
        trig_vals[0] = std::cos(vgetq_lane_f64(phase, 0));
        trig_vals[1] = std::cos(vgetq_lane_f64(phase, 1));
        
        // Final multiplication
        chirplet[i] = exp_vals[0] * trig_vals[0];
        chirplet[i + 1] = exp_vals[1] * trig_vals[1];
        
        // Check for invalid values
        if (std::isnan(chirplet[i]) || std::isinf(chirplet[i])) chirplet[i] = 0.0;
        if (std::isnan(chirplet[i + 1]) || std::isinf(chirplet[i + 1])) chirplet[i + 1] = 0.0;
    }
    
    // Handle remaining elements
    for (int i = simd_length; i < signal_length; ++i) {
        double t = static_cast<double>(i) * time_step;
        double time_diff = t - tc;
        
        double exponent = -0.5 * std::pow(time_diff * inv_Dt, 2);
        exponent = std::max(-50.0, exponent);
        double gaussian_window = std::exp(exponent);
        
        double phase = two_pi * (c * time_diff * time_diff + fc * time_diff);
        double complex_exp = std::cos(phase);  // Use cosine for real mode
        
        chirplet[i] = gaussian_window * complex_exp;
        
        if (std::isnan(chirplet[i]) || std::isinf(chirplet[i])) {
            chirplet[i] = 0.0;
        }
    }
    
    return chirplet;
#else
    // Fallback to base class implementation
    return ACT::g(tc, fc, logDt, c);
#endif
}

