#ifndef ACT_CUDA_TK_H
#define ACT_CUDA_TK_H

#include "ACT_CPU.h"
#include "ACT.h"
#include <utility>
#include <vector>
#include <memory>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

/**
 * ACT_CUDA_TK - Adaptive Chirplet Transform with custom CUDA kernels (ThunderKittens-ready)
 *
 * Notes:
 * - Single-signal dictionary search implemented with a custom CUDA GEMV + host argmax.
 * - Designed to be swapped with ThunderKittens kernels when headers are available.
 * - Double specialization falls back to CPU.
 */
template <typename Scalar>
class ACT_CUDA_TK_T : public ACT_CPU_T<Scalar> {
public:
    using Base = ACT_CPU_T<Scalar>;
    using ParameterRanges = typename Base::ParameterRanges;

    ACT_CUDA_TK_T(double FS,
                  int length,
                  const ParameterRanges& ranges,
                  bool verbose = false)
        : Base(FS, length, ranges, verbose) {
        if (verbose) {
            std::cout << "\n===============================================\n";
            std::cout << "INITIALIZING ACT_CUDA_TK" << std::endl;
            std::cout << "===============================================\n\n";
#ifdef USE_CUDA
            std::cout << "[ACT_CUDA_TK] Custom CUDA kernel path enabled for float32; double uses CPU fallback" << std::endl;
#else
            std::cout << "[ACT_CUDA_TK] Built without USE_CUDA; using CPU fallback" << std::endl;
#endif
            std::cout << std::endl;
        }
    }

    ACT_CUDA_TK_T(double FS,
                  int length,
                  const ACT::ParameterRanges& ranges,
                  bool verbose = false)
        : ACT_CUDA_TK_T(FS, length,
                        ParameterRanges(ranges.tc_min, ranges.tc_max, ranges.tc_step,
                                        ranges.fc_min, ranges.fc_max, ranges.fc_step,
                                        ranges.logDt_min, ranges.logDt_max, ranges.logDt_step,
                                        ranges.c_min, ranges.c_max, ranges.c_step),
                        verbose) {}

    ~ACT_CUDA_TK_T() override;

    using Base::search_dictionary;
    std::pair<int, Scalar> search_dictionary(const Eigen::Ref<const act::VecX<Scalar>>& signal) const override;

    int generate_chirplet_dictionary() override;
    void on_dictionary_loaded() override;

private:
#ifdef USE_CUDA
    mutable bool cuda_ready_ = false;
    mutable int m_ = 0;
    mutable int n_ = 0;
    mutable std::vector<float> dict_colmajor_f32_;

    // Device buffers
    mutable float* d_A_ = nullptr;      // (m_ x n_), column-major
    mutable float* d_x_ = nullptr;      // (m_)
    mutable float* d_scores_ = nullptr; // (n_)

    // Device-side argmax (two-stage reduction)
    // Per-block best (length = ceil(n_/TN_TILE), where TN_TILE=128)
    mutable float* d_blk_best_abs_ = nullptr; // abs value per block
    mutable float* d_blk_best_val_ = nullptr; // signed value per block
    mutable int*   d_blk_best_idx_ = nullptr; // index per block
    mutable int    blk_count_ = 0;            // number of per-block entries allocated

    // Final best written by tiny reducer (single element each)
    mutable float* d_final_best_val_ = nullptr; // length 1
    mutable int*   d_final_best_idx_ = nullptr; // length 1

    mutable cudaStream_t stream_ = nullptr;

    void ensure_cuda_resources() const;
    void ensure_cuda_dict() const;
    void warmup_kernels() const;
    void cleanup_cuda() const;
#endif
};

using ACT_CUDA_TK = ACT_CUDA_TK_T<double>;
using ACT_CUDA_TK_f = ACT_CUDA_TK_T<float>;

#endif // ACT_CUDA_TK_H
