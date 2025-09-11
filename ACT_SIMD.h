#ifndef ACT_SIMD_H
#define ACT_SIMD_H

#include "ACT.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/**
 * SIMD-optimized Adaptive Chirplet Transform
 * Optimized for Apple Silicon with Accelerate framework and ARM NEON
 * Falls back to scalar implementation on other platforms
 */
class ACT_SIMD : public ACT {
    friend class ACT_Benchmark; // Allow ACT_Benchmark to access protected members
public:
    /**
     * Constructor - same as base ACT class
     */
    ACT_SIMD(double FS, int length, 
             const ParameterRanges& ranges,
             bool complex_mode = false, bool verbose = false);
    virtual ~ACT_SIMD();

    /**
     * Destructor
     */
    //~ACT_SIMD();

    /**
     * SIMD-optimized dictionary search
     * @param signal Input signal
     * @return Pair of (best_index, best_value)
     */
    std::pair<int, double> search_dictionary(const std::vector<double>& signal) override;

    
    
    
protected:
    /**
     * SIMD-optimized inner product using Apple Accelerate framework
     * @param a First vector
     * @param b Second vector
     * @return Dot product result
     */
    double inner_product_accelerate(const std::vector<double>& a, const std::vector<double>& b);

    /**
     * SIMD-optimized inner product using ARM NEON intrinsics
     * @param a First vector (must be aligned)
     * @param b Second vector (must be aligned)
     * @param length Vector length (must be multiple of 2)
     * @return Dot product result
     */
    double inner_product_neon(const double* __restrict a, const double* __restrict b, size_t length);

    /**
     * Compiler auto-vectorized inner product
     * @param a First vector
     * @param b Second vector
     * @return Dot product result
     */
    double inner_product_auto_vectorized(const std::vector<double>& a, const std::vector<double>& b);

    /**
     * Batch SIMD dictionary search - process multiple chirplets simultaneously
     * @param signal Input signal
     * @param start_idx Starting dictionary index
     * @param batch_size Number of chirplets to process
     * @param results Output array for inner product results
     */
    void batch_inner_products_accelerate(const std::vector<double>& signal,
                                       int start_idx, int batch_size,
                                       std::vector<double>& results);

public:
    /**
     * SIMD-accelerated objective function for optimization
     * Override of base class minimize_this() with SIMD inner product
     * @param params Chirplet parameters [tc, fc, logDt, c]
     * @param signal Input signal
     * @return Negative inner product (for minimization)
     */
    double minimize_this(const std::vector<double>& params, const std::vector<double>& signal) override;

    /**
     * SIMD-accelerated chirplet generation for optimization only
     * Separate method to avoid interfering with dictionary generation
     * @param tc Time center parameter
     * @param fc Frequency center parameter
     * @param logDt Log duration parameter
     * @param c Chirp rate parameter
     * @return Generated chirplet vector
     */
    std::vector<double> g_simd_optimization(double tc, double fc, double logDt, double c);

protected:
    /**
     * SIMD-accelerated chirplet generation using Apple Accelerate
     * @param tc Time center parameter
     * @param fc Frequency center parameter
     * @param logDt Log duration parameter
     * @param c Chirp rate parameter
     * @return Generated chirplet vector
     */
    std::vector<double> g_accelerate(double tc, double fc, double logDt, double c);

    /**
     * SIMD-accelerated chirplet generation using ARM NEON
     * @param tc Time center parameter
     * @param fc Frequency center parameter
     * @param logDt Log duration parameter
     * @param c Chirp rate parameter
     * @return Generated chirplet vector
     */
    std::vector<double> g_neon(double tc, double fc, double logDt, double c);

private:
    // Runtime feature detection
    bool has_accelerate;
    bool has_neon;

    /**
     * Detect available SIMD features at runtime
     */
    void detect_simd_features();

    /**
     * Initialize aligned memory for SIMD operations
     */
    void initialize_simd_memory();

    /**
     * Cleanup aligned memory
     */
    void cleanup_simd_memory();
};

#endif // ACT_SIMD_H
