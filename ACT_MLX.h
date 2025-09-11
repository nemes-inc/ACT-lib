#ifndef ACT_MLX_H
#define ACT_MLX_H

#include "ACT.h"
#include <cstddef>
#include <utility>
#include <vector>

/**
 * ACT_MLX - Adaptive Chirplet Transform with a GPU-accelerated dictionary search (Apple MLX)
 *
 * This class scaffolds a drop-in backend that overrides search_dictionary() and will
 * call into MLX C++ when ACT_USE_MLX is defined at build time. By default (no MLX),
 * it safely falls back to the base-class implementation so existing builds keep working.
 */
class ACT_MLX : public ACT {
public:
    ACT_MLX(double FS,
            int length,
            const ParameterRanges& ranges,
            bool complex_mode = false,
            bool verbose = false);
    virtual ~ACT_MLX() = default;

    // GPU-accelerated dictionary search (fallbacks to CPU when MLX is disabled)
    std::pair<int, double> search_dictionary(const std::vector<double>& signal) override;

    // Optional knobs (no-ops unless/untill MLX path is enabled)
    void set_tile_size(std::size_t M) { tile_size = M; }
    void enable_mlx(bool enable = true) { use_mlx = enable; }
    void use_precomputed_gemv(bool enable = true) { prefer_gemv = enable; }

private:
    bool use_mlx;               // runtime toggle; also requires ACT_USE_MLX at compile time
    std::size_t tile_size;      // atoms per tile when generating on-the-fly on device
    bool prefer_gemv = true;    // CPU fallback: use BLAS GEMV on a flattened dictionary

    // CPU fallback storage for GEMV path
    std::vector<double> dict_flat;   // Row-major [dict_size, length]
    bool dict_flat_ready = false;
    std::vector<double> scores_buffer; // reuse allocations for output scores

    // Ensure dict_flat is materialized from dict_mat
    void ensure_flattened_dictionary();
};

#endif // ACT_MLX_H
