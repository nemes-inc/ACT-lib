#ifndef ACT_CUBLAS_H
#define ACT_CUBLAS_H

#include "ACT_CPU.h"
#include "ACT.h" // for ACT::ParameterRanges convenience conversion
#include <utility>
#include <vector>
#include <memory>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

/**
 * ACT_cuBLAS - Adaptive Chirplet Transform with a cuBLAS-accelerated dictionary search
 *
 * This backend overrides search_dictionary() to compute scores = A^T * x on the GPU
 * using cuBLAS for float32 specialization. Double specialization falls back to CPU.
 */
template <typename Scalar>
class ACT_cuBLAS_T : public ACT_CPU_T<Scalar> {
public:
    using Base = ACT_CPU_T<Scalar>;
    using ParameterRanges = typename Base::ParameterRanges;

    // Primary constructor using ACT_CPU parameter ranges
    ACT_cuBLAS_T(double FS,
                 int length,
                 const ParameterRanges& ranges,
                 bool verbose = false)
        : Base(FS, length, ranges, verbose) {
        if (verbose) {
            std::cout << "\n===============================================\n";
            std::cout << "INITIALIZING ACT_cuBLAS" << std::endl;
            std::cout << "===============================================\n\n";
#ifdef USE_CUDA
            std::cout << "[ACT_cuBLAS] cuBLAS path enabled for float32; double uses CPU fallback" << std::endl;
#else
            std::cout << "[ACT_cuBLAS] Built without USE_CUDA; using CPU fallback" << std::endl;
#endif
            std::cout << std::endl;
        }
    }

    // Convenience constructor to accept ACT::ParameterRanges and convert
    ACT_cuBLAS_T(double FS,
                 int length,
                 const ACT::ParameterRanges& ranges,
                 bool verbose = false)
        : ACT_cuBLAS_T(FS, length,
                        ParameterRanges(ranges.tc_min, ranges.tc_max, ranges.tc_step,
                                        ranges.fc_min, ranges.fc_max, ranges.fc_step,
                                        ranges.logDt_min, ranges.logDt_max, ranges.logDt_step,
                                        ranges.c_min, ranges.c_max, ranges.c_step),
                        verbose) {}

    ~ACT_cuBLAS_T() override;

    // Bring in base overloads
    using Base::search_dictionary;
    // Override Eigen-based search to enable cuBLAS path for float32
    std::pair<int, Scalar> search_dictionary(const Eigen::Ref<const act::VecX<Scalar>>& signal) const override;

    // Override dictionary generation to upload to device and warm up (float32 only)
    int generate_chirplet_dictionary() override;

    // When loading a dictionary from disk, perform device packing/warmup too
    void on_dictionary_loaded() override;

private:
#ifdef USE_CUDA
    // Lazily packed column-major dictionary buffer (float32 only)
    mutable bool cuda_ready_ = false;
    mutable int m_ = 0;
    mutable int n_ = 0;
    mutable std::vector<float> dict_colmajor_f32_; // size m_*n_

    // Device buffers
    mutable float* d_A_ = nullptr;      // (m_ x n_) column-major
    mutable float* d_x_ = nullptr;      // (m_)
    mutable float* d_scores_ = nullptr; // (n_)

    // cuBLAS handle + stream
    mutable cublasHandle_t handle_ = nullptr;
    mutable cudaStream_t stream_ = nullptr;

    void ensure_cuda_resources() const;
    void ensure_cuda_dict() const;
    void warmup_kernels() const;
    void cleanup_cuda() const;
#endif
};

// Default double-precision alias for compatibility
using ACT_cuBLAS = ACT_cuBLAS_T<double>;
using ACT_cuBLAS_f = ACT_cuBLAS_T<float>;

#endif // ACT_CUBLAS_H
