#include "ACT_MLX.h"
#include "ACT.h"

#include <iostream>
#include <limits>
#include <algorithm>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Future MLX includes will be guarded by ACT_USE_MLX
// #ifdef ACT_USE_MLX
// #include <mlx/...>  // MLX C++ headers TBD
// #endif

ACT_MLX::ACT_MLX(double FS,
                 int length,
                 const ParameterRanges& ranges,
                 bool complex_mode,
                 bool verbose)
    : ACT(FS, length, ranges, complex_mode, verbose),
      use_mlx(false),
      tile_size(16384) { // reasonable default, will be tuned
}

void ACT_MLX::ensure_flattened_dictionary() {
    if (dict_flat_ready) return;
    const int M = get_dict_size();
    const int N = get_length();
    dict_flat.assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0);
    const auto &dict = get_dict_mat();
    for (int i = 0; i < M; ++i) {
        const double *row = dict[i].data();
        std::copy(row, row + N, dict_flat.begin() + static_cast<size_t>(i) * N);
    }
    dict_flat_ready = true;
}

std::pair<int, double> ACT_MLX::search_dictionary(const std::vector<double>& signal) {
    // If runtime flag disabled, just call base implementation
    if (!use_mlx) {
        // CPU fallback path with optional GEMV acceleration
        if (prefer_gemv) {
            ensure_flattened_dictionary();

#ifdef __APPLE__
            if (verbose) {
                std::cout << "[ACT_MLX] Using GEMV for dictionary search\n";
            }
            const int M = get_dict_size();
            const int N = get_length();
            if (scores_buffer.size() != static_cast<size_t>(M)) {
                scores_buffer.assign(M, 0.0);
            }

            // y = A @ x (row-major). A: MxN, x: Nx1, y: Mx1
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        M, N,
                        1.0,
                        dict_flat.data(), N,
                        signal.data(), 1,
                        0.0,
                        scores_buffer.data(), 1);

            // Argmax
            int best_idx = 0;
            double best_val = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < M; ++i) {
                double v = scores_buffer[i];
                if (v > best_val) { best_val = v; best_idx = i; }
            }
            return {best_idx, best_val};
#else
            // Non-Apple fallback: simple loop
            const auto &dict = get_dict_mat();
            int best_idx = 0;
            double best_val = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < get_dict_size(); ++i) {
                // std::inner_product
                double sum = 0.0;
                const auto &row = dict[i];
                for (int j = 0; j < get_length(); ++j) sum += row[j] * signal[j];
                if (sum > best_val) { best_val = sum; best_idx = i; }
            }
            return {best_idx, best_val};
#endif
        }

        // Fall back to base class scalar search
        return ACT::search_dictionary(signal);
    }

#ifdef ACT_USE_MLX
    // TODO: Implement MLX-accelerated search (tiled on-the-fly generation or precomputed GEMV)
    // Placeholder: fall back for now so we can land the skeleton safely
    return ACT::search_dictionary(signal);
#else
    // Compiled without MLX; warn once and fall back
    static bool warned = false;
    if (!warned) {
        if (verbose) {
            std::cerr << "[ACT_MLX] Built without ACT_USE_MLX. Falling back to CPU search_dictionary.\n";
        }
        warned = true;
    }
    return ACT::search_dictionary(signal);
#endif
}
