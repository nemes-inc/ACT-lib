#ifndef ACT_H
#define ACT_H

#include <vector>
#include <complex>
#include <string>
#include <memory>
#include <fstream>
#include <istream>

// ALGLIB includes for optimization
#include "alglib/alglib-cpp/src/optimization.h"

/**
 * Adaptive Chirplet Transform Library - C++ Implementation
 * 
 * One-to-one translation of the Python ACT implementation
 * Maintains exact same functionality and algorithms:
 * - Dictionary-based matching pursuit
 * - BFGS optimization for parameter refinement
 * - Gaussian-enveloped chirp generation
 * - P-order signal decomposition
 */

class ACT {
    friend class ACTOptimizer;  // Allow ACTOptimizer to access private members
    friend class ACT_Benchmark; // Allow ACT_Benchmark to access protected members
public:
    // Parameter ranges structure
    struct ParameterRanges {
        double tc_min, tc_max, tc_step;      // Time center
        double fc_min, fc_max, fc_step;      // Frequency center  
        double logDt_min, logDt_max, logDt_step;  // Log duration
        double c_min, c_max, c_step;         // Chirp rate
        
        ParameterRanges(double tc_min = 0, double tc_max = 76, double tc_step = 1,
                       double fc_min = 0.7, double fc_max = 15, double fc_step = 0.2,
                       double logDt_min = -4, double logDt_max = -1, double logDt_step = 0.3,
                       double c_min = -30, double c_max = 30, double c_step = 3)
            : tc_min(tc_min), tc_max(tc_max), tc_step(tc_step),
              fc_min(fc_min), fc_max(fc_max), fc_step(fc_step),
              logDt_min(logDt_min), logDt_max(logDt_max), logDt_step(logDt_step),
              c_min(c_min), c_max(c_max), c_step(c_step) {}
    };

    // Transform result structure
    struct TransformResult {
        std::vector<std::vector<double>> params;  // [order x 4] parameter matrix
        std::vector<double> coeffs;               // coefficient list
        std::vector<double> signal;               // original signal
        double error;                             // residual error
        std::vector<double> residue;              // final residue
        std::vector<double> approx;               // approximation
    };

protected:
    // Core parameters
    double FS;                                    // Sampling frequency
    int length;                                   // Signal length
    ParameterRanges param_ranges;                 // Parameter ranges
    bool complex_mode;                            // Complex/real mode
    bool verbose;                                 // Verbose output control
    
    // Dictionary matrices
    std::vector<std::vector<double>> dict_mat;    // Dictionary matrix [dict_size x length]
    std::vector<std::vector<double>> param_mat;   // Parameter matrix [dict_size x 4]
    int dict_size;                                // Dictionary size
    
public:
    /**
     * Constructor - Initialize ACT with parameters
     */
    ACT(double FS, int length, 
        const ParameterRanges& ranges,
        bool complex_mode = false, bool verbose = false);
    
    /**
     * Destructor
     */
    ~ACT();

    /**
     * Generate a single chirplet (equivalent to Python g() method)
     * @param tc Time center (in samples)
     * @param fc Frequency center (in Hz)
     * @param logDt Log of duration
     * @param c Chirp rate (in Hz/s)
     * @return Chirplet signal vector
     */
    virtual std::vector<double> g(double tc = 0, double fc = 1, double logDt = 0, double c = 0);

    /**
     * Generate chirplet dictionary
     * @param debug Enable debug output
     * @return Dictionary size
     */
    int generate_chirplet_dictionary();

    /**
     * Save the current dictionary and all defining parameters to a file.
     * The file contains: format header + version, FS, length, complex_mode, ParameterRanges,
     * dictionary size, dictionary matrix and parameter matrix.
     * @param file_path Destination file path
     * @return true on success, false on failure
     */
    bool save_dictionary(const std::string& file_path) const;

    /**
     * Load a dictionary file and return a properly initialized ACT instance.
     * The returned instance will have its parameters set and dictionary matrices loaded from the file.
     * Usage: auto simd = ACT::load_dictionary<ACT_SIMD>(path);
     * @param file_path Source file path
     * @param mute Control console output for the created instance
     * @return unique_ptr to an ACT instance on success, nullptr on failure
     */
    template <typename TDerived>
    static std::unique_ptr<TDerived> load_dictionary(const std::string& file_path, bool verbose = false) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) return nullptr;
        // Create a placeholder instance; load_dictionary_data_from_stream will overwrite
        // FS, length, complex_mode, param_ranges, dict_size, dict_mat, and param_mat
        std::unique_ptr<TDerived> instance(new TDerived(128.0, 76, ParameterRanges(), /*complex_mode*/false, /*verbose*/verbose));
        if (!load_dictionary_data_from_stream(file, *instance)) {
            return nullptr;
        }
        return instance;
    }

    /**
     * Search dictionary for best matching chirplet
     * @param signal Input signal
     * @return Pair of (best_index, best_value)
     */
    virtual std::pair<int, double> search_dictionary(const std::vector<double>& signal);

    /**
     * Perform P-order ACT transform
     * @param signal Input signal
     * @param order Transform order (number of chirplets)
     * @param residual_threshold Residual threshold for early stopping
     * @param debug Enable debug output
     * @return Transform result structure
     */
    virtual TransformResult transform(const std::vector<double>& signal, int order = 5, double residual_threshold = 1e-6);

    /**
     * Get dictionary length
     * @return Dictionary length
     */
    virtual int get_dictionary_length() const { return length; };

    

    /**
     * Cost function for optimization (equivalent to minimize_this)
     * @param params Parameter vector [tc, fc, logDt, c]
     * @param signal Target signal
     * @return Negative inner product (to be minimized)
     */
    virtual double minimize_this(const std::vector<double>& params, const std::vector<double>& signal);

    // Utility functions
    int get_dict_size() const { return dict_size; }
    double get_FS() const { return FS; }
    int get_length() const { return length; }
    const ParameterRanges& get_param_ranges() const { return param_ranges; }
    double inner_product(const std::vector<double>& a, const std::vector<double>& b);

    // Diagnostic access methods
    const std::vector<std::vector<double>>& get_dict_mat() const { return dict_mat; }
    const std::vector<std::vector<double>>& get_param_mat() const { return param_mat; }
    bool get_complex_mode() const { return complex_mode; }

protected:
    // Protected methods for inheritance
    std::vector<double> bfgs_optimize(const std::vector<double>& initial_params, 
                                      const std::vector<double>& signal);
    
    /**
     * Helper used by subclass loaders to populate an existing instance from a stream.
     * Expects the file pointer to be at the beginning of the file.
     * This will overwrite FS, length, complex_mode, param_ranges, dict_size, dict_mat, param_mat.
     * Returns true on success.
     */
    static bool load_dictionary_data_from_stream(std::istream& in, ACT& instance);
    
private:
    // Internal helper functions
    std::vector<double> linspace(double start, double end, double step);
    
    // Simple fallback optimization when ALGLIB fails
    std::vector<double> simple_optimize(const std::vector<double>& initial_params, 
                                       const std::vector<double>& signal);
};

// Optimization wrapper class for ALGLIB
class ACTOptimizer {    
public:
    ACTOptimizer(ACT* act_instance, const std::vector<double>& signal) : act_instance(act_instance), signal(signal) {}
    static void optimization_function(const alglib::real_1d_array &x, double &func, void *ptr);
    static void optimization_function_with_gradient(const alglib::real_1d_array &x, double &func, alglib::real_1d_array &grad, void *ptr);
    ACT* act_instance;
    std::vector<double> signal;
};

#endif // ACT_H
