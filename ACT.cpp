#include "ACT.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cassert>

// For BFGS optimization - using ALGLIB library
#include <functional>

ACT::ACT(double FS, int length, const std::string& dict_addr, 
         const ParameterRanges& ranges, bool complex_mode, 
         bool force_regenerate, bool mute)
    : FS(FS), length(length), param_ranges(ranges), complex_mode(complex_mode), mute(mute), is_debug(false),
      dict_cache_file(dict_addr), force_regenerate(force_regenerate), dict_size(0) {
    
    if (!mute) {
        std::cout << "\n===============================================\n";
        std::cout << "INITIALIZING ADAPTIVE CHIRPLET TRANSFORM MODULE\n";
        std::cout << "===============================================\n\n";
    }

    // Try to load cached dictionary first
    if (!force_regenerate && load_dictionary_cache()) {
        if (!mute) std::cout << "Found Chirplet Dictionary, Loading File...\n";
    } else {
        if (!mute) std::cout << "Did not find cached chirplet dictionary matrix, generating chirplet dictionary...\n\n";
        generate_chirplet_dictionary(true);
        if (!mute) std::cout << "\nDone Generating Chirplet Dictionary\n";
        
        if (!mute) std::cout << "\nCaching Generated Dictionary/Parameter Matrices...\n";
        save_dictionary_cache();
        if (!mute) std::cout << "Done Caching.\n";
    }

    if (!mute) {
        std::cout << "=====================================================\n";
        std::cout << "DONE INITIALIZING ADAPTIVE CHIRPLET TRANSFORM MODULE.\n";
        std::cout << "=====================================================\n";
    }
}

ACT::~ACT() {
    // Cleanup if needed
}

std::vector<double> ACT::g(double tc, double fc, double logDt, double c) {
    // Add bounds checking for numerical stability
    if (std::isnan(tc) || std::isnan(fc) || std::isnan(logDt) || std::isnan(c)) {
        return std::vector<double>(length, 0.0);
    }
    
    // Clamp logDt to prevent extreme values
    logDt = std::max(-10.0, std::min(2.0, logDt));
    
    // Convert tc from samples to seconds (matching Python implementation)
    tc /= FS;
    
    // Calculate duration from log value with bounds checking
    double Dt = std::exp(logDt);
    if (Dt < 1e-10 || Dt > 100.0) {
        return std::vector<double>(length, 0.0);
    }
    
    // Create time array
    std::vector<double> t(length);
    for (int i = 0; i < length; ++i) {
        t[i] = static_cast<double>(i) / FS;
    }
    
    // Generate chirplet components
    std::vector<double> chirplet(length);
    for (int i = 0; i < length; ++i) {
        double time_diff = t[i] - tc;
        
        // Gaussian window with numerical stability
        double exponent = -0.5 * std::pow(time_diff / Dt, 2);
        if (exponent < -50.0) {
            exponent = -50.0;  // Prevent underflow
        }
        double gaussian_window = std::exp(exponent);
        
        // Complex exponential (taking real part for now)
        double phase = 2.0 * M_PI * (c * time_diff * time_diff + fc * time_diff);
        double complex_exp = complex_mode ? std::sin(phase) : std::cos(phase);
        
        chirplet[i] = gaussian_window * complex_exp;
        
        // Check for invalid values
        if (std::isnan(chirplet[i]) || std::isinf(chirplet[i])) {
            chirplet[i] = 0.0;
        }
    }
    // L2 normalize chirplet to unit energy to avoid duration bias
    double energy = 0.0;
    for (int i = 0; i < length; ++i) energy += chirplet[i] * chirplet[i];
    if (energy > 0.0) {
        double inv_norm = 1.0 / std::sqrt(energy);
        for (int i = 0; i < length; ++i) chirplet[i] *= inv_norm;
    } else {
        // Fallback: return zeros if degenerate
        std::fill(chirplet.begin(), chirplet.end(), 0.0);
    }

    return chirplet;
}

int ACT::generate_chirplet_dictionary(bool debug) {
    // Generate parameter value arrays
    auto tc_vals = linspace(param_ranges.tc_min, param_ranges.tc_max, param_ranges.tc_step);
    auto fc_vals = linspace(param_ranges.fc_min, param_ranges.fc_max, param_ranges.fc_step);
    auto logDt_vals = linspace(param_ranges.logDt_min, param_ranges.logDt_max, param_ranges.logDt_step);
    auto c_vals = linspace(param_ranges.c_min, param_ranges.c_max, param_ranges.c_step);
    
    dict_size = tc_vals.size() * fc_vals.size() * logDt_vals.size() * c_vals.size();
    
    if (debug) {
        std::cout << "Dictionary length: " << dict_size << std::endl;
        std::cout << "Parameter ranges:\n";
        std::cout << "  tc: " << tc_vals.size() << " values (" << param_ranges.tc_min << " to " << param_ranges.tc_max << ")\n";
        std::cout << "  fc: " << fc_vals.size() << " values (" << param_ranges.fc_min << " to " << param_ranges.fc_max << ")\n";
        std::cout << "  logDt: " << logDt_vals.size() << " values (" << param_ranges.logDt_min << " to " << param_ranges.logDt_max << ")\n";
        std::cout << "  c: " << c_vals.size() << " values (" << param_ranges.c_min << " to " << param_ranges.c_max << ")\n";
    }
    
    // Initialize matrices
    dict_mat.resize(dict_size, std::vector<double>(length));
    param_mat.resize(dict_size, std::vector<double>(4));
    
    int cnt = 0;
    int slow_cnt = 1;
    
    for (double tc : tc_vals) {
        if (debug) {
            std::cout << "\n" << slow_cnt << "/" << tc_vals.size() << ": \t";
            slow_cnt++;
        }
        
        for (double fc : fc_vals) {
            if (debug) std::cout << ".";
            
            for (double logDt : logDt_vals) {
                for (double c : c_vals) {
                    // Generate chirplet
                    dict_mat[cnt] = g(tc, fc, logDt, c);
                    
                    // Store parameters
                    param_mat[cnt][0] = tc;
                    param_mat[cnt][1] = fc;
                    param_mat[cnt][2] = logDt;
                    param_mat[cnt][3] = c;
                    
                    cnt++;
                }
            }
        }
    }
    
    return dict_size;
}

std::pair<int, double> ACT::search_dictionary(const std::vector<double>& signal) {
    assert(signal.size() == length);
    
    int best_idx = 0;
    double best_val = -std::numeric_limits<double>::infinity();
    
    for (int i = 0; i < dict_size; ++i) {
        double val = inner_product(dict_mat[i], signal);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    
    return std::make_pair(best_idx, best_val);
}

ACT::TransformResult ACT::transform(const std::vector<double>& signal, int order, double residual_threshold, bool debug) {
    this->is_debug = debug;
    TransformResult result;
    result.params.resize(order, std::vector<double>(4));
    result.coeffs.resize(order);
    result.signal = signal;
    result.approx.resize(length, 0.0);
    result.residue = signal;  // Copy signal to residue
    
    if (debug) {
        std::cout << "Beginning " << order << "-Order Transform of Input Signal...\n";
    }

    double prev_resid_norm2 = std::numeric_limits<double>::max();
    
    for (int i = 0; i < order; ++i) {
        if (debug) std::cout << ".";

        // Find best matching chirplet from dictionary
        auto [ind, val] = search_dictionary(result.residue);

        // Get coarse parameters
        std::vector<double> params = param_mat[ind];

        // Fine-tune parameters using BFGS optimization
        std::vector<double> refined_params = bfgs_optimize(params, result.residue);
        if (this->is_debug) {
            std::cout << std::fixed << std::setprecision(8);
            std::cout << "[DEBUG] bfgs_optimize returned refined_params: "
                      << "tc=" << refined_params[0] << ", "
                      << "fc=" << refined_params[1] << ", "
                      << "logDt=" << refined_params[2] << ", "
                      << "c=" << refined_params[3] << std::endl;
        }

        // Generate optimized chirplet
        auto updated_base_chirplet = g(refined_params[0], refined_params[1], refined_params[2], refined_params[3]);

        // Calculate coefficient (unit-energy atoms => LS amplitude is simple dot product)
        double updated_chirplet_coeff = inner_product(updated_base_chirplet, result.residue);

        // Store results for this order
        result.params[i] = refined_params;
        result.coeffs[i] = updated_chirplet_coeff;

        // Create new chirp component
        std::vector<double> new_chirp(length);
        for (int j = 0; j < length; ++j) {
            new_chirp[j] = updated_base_chirplet[j] * updated_chirplet_coeff;
        }

        // Update residue and approximation
        double resid_norm2 = 0.0;
        for (int j = 0; j < length; ++j) {
            result.residue[j] -= new_chirp[j];
            result.approx[j] += new_chirp[j];
            resid_norm2 += result.residue[j] * result.residue[j];
        }
        if (this->is_debug) {
            std::cout << "[DEBUG] Residual norm2: " << resid_norm2 << std::endl;
        }

        // Check for early stopping
        if (prev_resid_norm2 - resid_norm2 < residual_threshold) {
            if (debug) std::cout << "Early stopping at order " << i << std::endl;
            break;
        }

        prev_resid_norm2 = resid_norm2;
    }
    
    if (debug) std::cout << std::endl;
    
    // Calculate final error
    double resid_norm2 = 0.0;
    for (int j = 0; j < length; ++j) resid_norm2 += result.residue[j] * result.residue[j];
    result.error = std::sqrt(resid_norm2); // residual L2 norm
    
    return result;
}

double ACT::minimize_this(const std::vector<double>& params, const std::vector<double>& signal) {
    // Check for invalid parameters
    for (double param : params) {
        if (std::isnan(param) || std::isinf(param)) {
            return 1e10;  // Large penalty for invalid parameters
        }
    }
    
    auto atom = g(params[0], params[1], params[2], params[3]);
    
    // Check if atom generation failed
    if (atom.empty()) {
        return 1e10;
    }
    
    double product = inner_product(atom, signal);
    
    // Check for invalid inner product
    if (std::isnan(product) || std::isinf(product)) {
        return 1e10;
    }
    
    double result = -std::abs(product);
    
    // Final check for invalid result
    if (std::isnan(result) || std::isinf(result)) {
        return 1e10;
    }
    
    return result;
}

// Helper functions
std::vector<double> ACT::linspace(double start, double end, double step) {
    std::vector<double> result;
    for (double val = start; val <= end; val += step) {
        result.push_back(val);
    }
    return result;
}

// Store signal for use in optimization function
std::vector<double> optimization_signal;

// ACTOptimizer function implementations
void ACTOptimizer::optimization_function(const alglib::real_1d_array &x, double &func, void *ptr) {
    // Convert ALGLIB array to std::vector
    std::vector<double> params(4);
    for (int i = 0; i < 4; ++i) {
        params[i] = x[i];
    }
    
    // Call the cost function
    ACTOptimizer* optimizer = static_cast<ACTOptimizer*>(ptr);
    std::vector<double>& signal = optimizer->signal;
    ACT* act_instance = optimizer->act_instance;
    func = act_instance->minimize_this(params, signal);
}

void ACTOptimizer::optimization_function_with_gradient(const alglib::real_1d_array &x, double &func, alglib::real_1d_array &grad, void *ptr) {
    // Convert ALGLIB array to std::vector
    std::vector<double> params(4);
    for (int i = 0; i < 4; ++i) {
        params[i] = x[i];
    }
    
    // Call the cost function
    ACTOptimizer* optimizer = static_cast<ACTOptimizer*>(ptr);
    std::vector<double>& signal = optimizer->signal;
    ACT* act_instance = optimizer->act_instance;
    func = act_instance->minimize_this(params, signal);
    
    // Check for invalid function value
    if (std::isnan(func) || std::isinf(func)) {
        func = 1e10;  // Large penalty for invalid values
    }
    
    // Define parameter bounds for gradient computation
    std::vector<double> lower_bounds = {0, 0.1, -5, -50};
    std::vector<double> upper_bounds = {static_cast<double>(act_instance->length - 1), 
                                       act_instance->FS / 2, 0, 50};
    
    // Compute gradient using finite differences with bounds checking
    double eps = 1e-6;
    std::vector<double> grad_vec(4);
    
    for (int i = 0; i < 4; ++i) {
        std::vector<double> params_plus = params;
        std::vector<double> params_minus = params;
        
        // Adjust step size to stay within bounds
        double step_plus = eps;
        double step_minus = eps;
        
        if (params[i] + eps > upper_bounds[i]) {
            step_plus = upper_bounds[i] - params[i];
        }
        if (params[i] - eps < lower_bounds[i]) {
            step_minus = params[i] - lower_bounds[i];
        }
        
        params_plus[i] += step_plus;
        params_minus[i] -= step_minus;
        
        double func_plus = act_instance->minimize_this(params_plus, signal);
        double func_minus = act_instance->minimize_this(params_minus, signal);
        
        // Check for invalid function values
        if (std::isnan(func_plus) || std::isinf(func_plus)) {
            func_plus = 1e10;
        }
        if (std::isnan(func_minus) || std::isinf(func_minus)) {
            func_minus = 1e10;
        }
        
        // Compute gradient with adaptive step size
        double total_step = step_plus + step_minus;
        if (total_step > 1e-12) {
            grad_vec[i] = (func_plus - func_minus) / total_step;
        } else {
            grad_vec[i] = 0.0;
        }
        
        // Check for invalid gradient values
        if (std::isnan(grad_vec[i]) || std::isinf(grad_vec[i])) {
            grad_vec[i] = 0.0;
        }
    }
    
    // Convert gradient to ALGLIB array
    grad.setlength(4);
    for (int i = 0; i < 4; ++i) {
        grad[i] = grad_vec[i];
    }
}


std::vector<double> ACT::bfgs_optimize(const std::vector<double>& initial_params, 
                                      const std::vector<double>& signal) {
    if (this->is_debug) {
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "[DEBUG] bfgs_optimize called with coarse params: "
                  << "tc=" << initial_params[0] << ", "
                  << "fc=" << initial_params[1] << ", "
                  << "logDt=" << initial_params[2] << ", "
                  << "c=" << initial_params[3] << std::endl;
    }
    // Store signal and instance for optimization function
    ACTOptimizer optimizer(this, signal);
    
    // Try ALGLIB numerical gradient optimization first
    try {
        alglib::minbcstate state;
        alglib::minbcreport rep;
        alglib::real_1d_array x;
        alglib::real_1d_array bndl;
        alglib::real_1d_array bndu;
        
        // Initialize parameter vector
        x.setlength(4);
        for (int i = 0; i < 4; ++i) {
            x[i] = initial_params[i];
        }
        
        // Set local parameter bounds: ±1 grid step around initial dictionary match,
        // clamped to the dictionary parameter ranges
        bndl.setlength(4);
        bndu.setlength(4);

        double tc_step = param_ranges.tc_step;
        double fc_step = param_ranges.fc_step;
        double logDt_step = param_ranges.logDt_step;
        double c_step = param_ranges.c_step;

        bndl[0] = std::max(param_ranges.tc_min,     initial_params[0] - tc_step);
        bndu[0] = std::min(param_ranges.tc_max,     initial_params[0] + tc_step);

        bndl[1] = std::max(param_ranges.fc_min,     initial_params[1] - fc_step);
        bndu[1] = std::min(param_ranges.fc_max,     initial_params[1] + fc_step);

        bndl[2] = std::max(param_ranges.logDt_min,  initial_params[2] - logDt_step);
        bndu[2] = std::min(param_ranges.logDt_max,  initial_params[2] + logDt_step);

        bndl[3] = std::max(param_ranges.c_min,      initial_params[3] - c_step);
        bndu[3] = std::min(param_ranges.c_max,      initial_params[3] + c_step);
        
        // Validate and clamp initial parameters to bounds
        for (int i = 0; i < 4; ++i) {
            if (x[i] < bndl[i]) x[i] = bndl[i] + 1e-6;
            if (x[i] > bndu[i]) x[i] = bndu[i] - 1e-6;
        }
        
        if (this->is_debug) {
            std::cout << "[DEBUG] Initial Guess (x0): "
                      << "tc=" << x[0] << ", "
                      << "fc=" << x[1] << ", "
                      << "logDt=" << x[2] << ", "
                      << "c=" << x[3] << std::endl;
            std::cout << "[DEBUG] Lower Bounds (bndl): "
                      << "tc=" << bndl[0] << ", "
                      << "fc=" << bndl[1] << ", "
                      << "logDt=" << bndl[2] << ", "
                      << "c=" << bndl[3] << std::endl;
            std::cout << "[DEBUG] Upper Bounds (bndu): "
                      << "tc=" << bndu[0] << ", "
                      << "fc=" << bndu[1] << ", "
                      << "logDt=" << bndu[2] << ", "
                      << "c=" << bndu[3] << std::endl;
        }

        // Create optimizer with numerical gradients (safer)
        alglib::minbccreatef(x, 1e-6, state);  // Use numerical gradients
        alglib::minbcsetbc(state, bndl, bndu);
        alglib::minbcsetcond(state, 1e-5, 0, 0, 100); // Increased maxits
        alglib::minbcsetxrep(state, false);
        
        // Optimize with numerical gradients only
        if (this->is_debug) {
            std::cout << "[DEBUG] Starting ALGLIB optimization..." << std::endl;
        }
        alglib::minbcoptimize(state, optimizer.optimization_function, nullptr, &optimizer);
        if (this->is_debug) {
            std::cout << "[DEBUG] ALGLIB optimization complete." << std::endl;
        }
        alglib::minbcresults(state, x, rep);
        if (this->is_debug) {
            std::cout << "[DEBUG] ALGLIB terminationtype=" << rep.terminationtype << std::endl;
        }
        
        // Check if optimization succeeded
        if (rep.terminationtype > 0) {
            std::vector<double> result(4);
            for (int i = 0; i < 4; ++i) {
                result[i] = x[i];
            }
            return result;
        } else {
            // Log termination type for diagnostics before falling back
            std::cerr << "ALGLIB terminationtype=" << rep.terminationtype
                      << ", falling back to simple optimization..." << std::endl;
        }
        
    } catch (const alglib::ap_error& e) {
        std::cerr << "ALGLIB error during optimization, falling back to simple optimization" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception during ALGLIB optimization: " << e.what() << std::endl;
    }
    
    // Fallback to simple gradient descent if ALGLIB fails
    std::cerr << "Using fallback simple optimization..." << std::endl;
    return simple_optimize(initial_params, signal);
}

std::vector<double> ACT::simple_optimize(const std::vector<double>& initial_params, 
                                        const std::vector<double>& signal) {
    std::vector<double> params = initial_params;
    double current_cost = minimize_this(params, signal);
    
    // Simple gradient descent with bounds checking
    double learning_rate = 0.01;
    int max_iterations = 20;
    
    // Parameter bounds
    std::vector<double> lower_bounds = {0, 0.1, -5, -50};
    std::vector<double> upper_bounds = {static_cast<double>(length - 1), FS / 2, 0, 50};
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<double> gradient(4);
        double eps = 1e-6;
        
        // Compute numerical gradient
        for (int i = 0; i < 4; ++i) {
            std::vector<double> params_plus = params;
            std::vector<double> params_minus = params;
            
            params_plus[i] += eps;
            params_minus[i] -= eps;
            
            double cost_plus = minimize_this(params_plus, signal);
            double cost_minus = minimize_this(params_minus, signal);
            
            gradient[i] = (cost_plus - cost_minus) / (2 * eps);
        }
        
        // Update parameters with gradient descent
        std::vector<double> new_params = params;
        for (int i = 0; i < 4; ++i) {
            new_params[i] -= learning_rate * gradient[i];
            
            // Apply bounds
            if (new_params[i] < lower_bounds[i]) new_params[i] = lower_bounds[i];
            if (new_params[i] > upper_bounds[i]) new_params[i] = upper_bounds[i];
        }
        
        double new_cost = minimize_this(new_params, signal);
        
        // Accept if improvement
        if (new_cost < current_cost) {
            params = new_params;
            current_cost = new_cost;
        } else {
            learning_rate *= 0.5;  // Reduce learning rate
            if (learning_rate < 1e-8) break;
        }
    }
    
    return params;
}

double ACT::inner_product(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

bool ACT::load_dictionary_cache() {
    std::ifstream file(dict_cache_file, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Read dictionary size
    file.read(reinterpret_cast<char*>(&dict_size), sizeof(dict_size));
    
    // Read dictionary matrix
    dict_mat.resize(dict_size, std::vector<double>(length));
    for (int i = 0; i < dict_size; ++i) {
        file.read(reinterpret_cast<char*>(dict_mat[i].data()), length * sizeof(double));
    }
    
    // Read parameter matrix
    param_mat.resize(dict_size, std::vector<double>(4));
    for (int i = 0; i < dict_size; ++i) {
        file.read(reinterpret_cast<char*>(param_mat[i].data()), 4 * sizeof(double));
    }
    
    return file.good();
}

void ACT::save_dictionary_cache() {
    std::ofstream file(dict_cache_file, std::ios::binary);
    if (!file.is_open()) return;
    
    // Write dictionary size
    file.write(reinterpret_cast<const char*>(&dict_size), sizeof(dict_size));
    
    // Write dictionary matrix
    for (int i = 0; i < dict_size; ++i) {
        file.write(reinterpret_cast<const char*>(dict_mat[i].data()), length * sizeof(double));
    }
    
    // Write parameter matrix
    for (int i = 0; i < dict_size; ++i) {
        file.write(reinterpret_cast<const char*>(param_mat[i].data()), 4 * sizeof(double));
    }
}
