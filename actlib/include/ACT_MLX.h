#ifndef ACT_MLX_H
#define ACT_MLX_H

#include "ACT_Accelerate.h"
#include "ACT.h" // for ACT::ParameterRanges convenience conversion
#include <utility>
#include <vector>
#include <memory>
#include <functional>

#ifdef USE_MLX
#include "mlx/mlx.h"
namespace mx = mlx::core;
#endif

/**
 * ACT_MLX - Adaptive Chirplet Transform with a GPU-accelerated dictionary search (Apple MLX)
 *
 * This class scaffolds a drop-in backend that overrides search_dictionary() and will
 * call into MLX C++ when integrated. For now it inherits the Accelerate-optimized
 * CPU path from ACT_Accelerate. No compile-time toggles are needed: selecting this
 * class selects the MLX-capable backend.
 */
template <typename Scalar>
class ACT_MLX_T : public ACT_Accelerate_T<Scalar> {
public:
    using Base = ACT_Accelerate_T<Scalar>;
    using ParameterRanges = typename Base::ParameterRanges;

    // Primary constructor using ACT_CPU/ACT_Accelerate parameter ranges
    ACT_MLX_T(double FS,
              int length,
              const ParameterRanges& ranges,
              bool verbose = false);

    // Convenience constructor to accept ACT::ParameterRanges and convert
    ACT_MLX_T(double FS,
              int length,
              const ACT::ParameterRanges& ranges,
              bool verbose = false);

    ~ACT_MLX_T() override = default;

    // Bring in base overloads
    using Base::search_dictionary;
    // Override Eigen-based search to enable MLX path later
    std::pair<int, Scalar> search_dictionary(const Eigen::Ref<const act::VecX<Scalar>>& signal) const override;

    // Override dictionary generation to pre-pack and warm up MLX (float32 only)
    int generate_chirplet_dictionary() override;

public:
    // When loading a dictionary from disk, perform MLX packing/warmup too
    void on_dictionary_loaded() override;

    // Expose device dictionary for batched GEMM paths (float32 only)
#ifdef USE_MLX
    const mx::array& get_dict_gpu() const;
#endif

private:
#ifdef USE_MLX
    // Lazily packed row-major dictionary and device array (float32 only)
    mutable bool mlx_ready_ = false;
    mutable std::vector<float> dict_rowmajor_; // shape [m * n] row-major
    mutable std::unique_ptr<mx::array> dict_gpu_; // shape {m, n}

    void ensure_mlx_dict() const;

    // Compiled MLX function: takes [A, x] and returns [idx, best_val]
    mutable std::function<std::vector<mx::array>(const std::vector<mx::array>&)> search_fn_;
#endif
};

// Default double-precision alias for compatibility
using ACT_MLX = ACT_MLX_T<double>;
using ACT_MLX_f = ACT_MLX_T<float>;

#endif // ACT_MLX_H
