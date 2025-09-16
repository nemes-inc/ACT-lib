#ifndef ACT_CPU_H
#define ACT_CPU_H

#include <vector>
#include <string>
#include <memory>
#include <istream>
#include <fstream>

#include "Eigen/Dense"
#include "act_types.h"

// ALGLIB forward include only for declarations in .cpp
// Keep header light to avoid heavy includes here

/**
 * Adaptive Chirplet Transform - CPU baseline using Eigen + CBLAS
 *
 * This class is independent from ACT (std::vector-based) and serves as the
 * foundation for all optimized CPU/GPU variants. It uses Eigen for math and
 * calls into BLAS (GEMV/DDOT/DAXPY) for core performance-sensitive paths.
 *
 * Notes:
 * - Dictionary file format is kept compatible with ACTDICT v2.
 * - Complex mode is dropped; only real-valued chirplets are generated.
 */
template <typename Scalar>
class ACT_CPU_T {
public:
    struct TransformOptions {
        int order = 5;
        double residual_threshold = 1e-6;
        bool refine = true; // if false, skip BFGS (coarse-only)
    };

    struct ParameterRanges {
        double tc_min, tc_max, tc_step;      // Time center (in samples)
        double fc_min, fc_max, fc_step;      // Frequency center (Hz)
        double logDt_min, logDt_max, logDt_step;  // Log duration
        double c_min, c_max, c_step;         // Chirp rate (Hz/s)

        ParameterRanges(double tc_min = 0, double tc_max = 76, double tc_step = 1,
                        double fc_min = 0.7, double fc_max = 15, double fc_step = 0.2,
                        double logDt_min = -4, double logDt_max = -1, double logDt_step = 0.3,
                        double c_min = -30, double c_max = 30, double c_step = 3)
            : tc_min(tc_min), tc_max(tc_max), tc_step(tc_step),
              fc_min(fc_min), fc_max(fc_max), fc_step(fc_step),
              logDt_min(logDt_min), logDt_max(logDt_max), logDt_step(logDt_step),
              c_min(c_min), c_max(c_max), c_step(c_step) {}
    };

    struct TransformResultT {
        // Rows = order, Cols = 4 (tc, fc, logDt, c)
        act::ParamsMat<Scalar> params;
        act::VecX<Scalar> coeffs;
        act::VecX<Scalar> signal;
        Scalar error = Scalar(0);
        act::VecX<Scalar> residue;
        act::VecX<Scalar> approx;
    };
    using TransformResult = TransformResultT;

public:
    ACT_CPU_T(double FS, int length,
              const ParameterRanges& ranges,
              bool verbose = false);
    virtual ~ACT_CPU_T();

    // Single chirplet atom (real-valued)
    virtual act::VecX<Scalar> g(double tc, double fc, double logDt, double c) const;

    // Build dictionary and parameter matrices. Returns dict_size.
    virtual int generate_chirplet_dictionary();

    // Dictionary search using BLAS GEMV: scores = A^T * signal
    virtual std::pair<int,Scalar> search_dictionary(const Eigen::Ref<const act::VecX<Scalar>>& signal) const;
    // Convenience adapter for std::vector without copy
    std::pair<int,Scalar> search_dictionary(const std::vector<Scalar>& signal) const;

    // P-order transform
    virtual TransformResultT transform(const Eigen::Ref<const act::VecX<Scalar>>& signal,
                                       int order = 5, double residual_threshold = 1e-6) const;
    virtual TransformResultT transform(const std::vector<Scalar>& signal,
                                       int order = 5, double residual_threshold = 1e-6) const;

    // Transform with options (supports coarse-only via refine=false)
    virtual TransformResultT transform(const Eigen::Ref<const act::VecX<Scalar>>& signal,
                                       const TransformOptions& options) const;

    // Objective for optimizer (negative inner product)
    double minimize_this(const std::vector<double>& params,
                         const Eigen::Ref<const act::VecX<Scalar>>& signal) const;

    // Save/Load dictionary (ACTDICT v2 compatible)
    bool save_dictionary(const std::string& file_path) const;

    template <typename TDerived>
    static std::unique_ptr<TDerived> load_dictionary(const std::string& file_path, bool verbose = false) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) return nullptr;
        // Use TDerived's own ParameterRanges type to avoid template-mismatch
        std::unique_ptr<TDerived> instance(new TDerived(128.0, 76, typename TDerived::ParameterRanges(), /*verbose*/verbose));
        if (!load_dictionary_data_from_stream(file, *instance)) {
            return nullptr;
        }
        // Allow derived backends (e.g., MLX) to perform device packing/warmup after load
        instance->on_dictionary_loaded();
        return instance;
    }

    // Hook invoked after dictionary is loaded from file; default no-op.
    // Public so factory/loader can invoke it on any derived backend (e.g., MLX)
    virtual void on_dictionary_loaded() {}

    // Accessors
    int get_dict_size() const { return dict_size; }
    double get_FS() const { return FS; }
    int get_length() const { return length; }
    const ParameterRanges& get_param_ranges() const { return param_ranges; }

    const act::MatX<Scalar>& get_dict_mat() const { return dict_mat; }
    const Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>& get_param_mat() const { return param_mat; }

private:
    // Internal helpers
    static bool load_dictionary_data_from_stream(std::istream& in, ACT_CPU_T<Scalar>& instance);

    Eigen::VectorXd linspace(double start, double end, double step) const;

    Eigen::VectorXd bfgs_optimize(const Eigen::Vector4d& initial_params,
                                  const Eigen::Ref<const act::VecX<Scalar>>& signal) const;

protected:
    act::VecX<Scalar> time_vector_seconds() const; // [0, 1/FS, 2/FS, ...]
    // Hooks for subclasses to override low-level math
    virtual Scalar dot(const Scalar* a, const Scalar* b, int n) const;
    virtual void axpy(int n, Scalar alpha, const Scalar* x, int incx, Scalar* y, int incy) const;

private:
    // Core parameters
    double FS;          // Sampling frequency
    int length;         // Signal length
    ParameterRanges param_ranges;
    bool verbose;

    // Dictionary as (length x dict_size), column-major (Eigen default)
    act::MatX<Scalar> dict_mat;
    // Parameters as (dict_size x 4)
    Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> param_mat;
    int dict_size = 0;
};

// Default double-precision alias for existing code
using ACT_CPU = ACT_CPU_T<double>;
using ACT_CPU_f = ACT_CPU_T<float>;

#endif // ACT_CPU_H
