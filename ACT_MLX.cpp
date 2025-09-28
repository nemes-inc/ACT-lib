#include "ACT_MLX.h"

#include <iostream>
#include <limits>
#include <algorithm>
#include <type_traits>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef USE_MLX
// Public accessor for device dictionary (float32 only)
template <typename Scalar>
const mx::array& ACT_MLX_T<Scalar>::get_dict_gpu() const {
    // Only meaningful for float specialization
    if constexpr (!std::is_same_v<Scalar, float>) {
        throw std::runtime_error("ACT_MLX_T<Scalar>::get_dict_gpu is only available for float specialization");
    }
    ensure_mlx_dict();
    return *dict_gpu_;
}
#endif

template <typename Scalar>
ACT_MLX_T<Scalar>::ACT_MLX_T(double FS,
                 int length,
                 const ParameterRanges& ranges,
                 bool verbose)
    : Base(FS, length, ranges, verbose) {
    if (verbose) {
        std::cout << "\n===============================================\n";
        std::cout << "INITIALIZING ACT_MLX\n";
        std::cout << "===============================================\n";
        std::cout << "[ACT_MLX] MLX search is enabled for float32 when USE_MLX=1; double falls back to Accelerate CPU path" << std::endl;
        std::cout << std::endl;
    }
}

template <typename Scalar>
ACT_MLX_T<Scalar>::ACT_MLX_T(double FS,
                 int length,
                 const ACT::ParameterRanges& ranges,
                 bool verbose)
    : ACT_MLX_T(
        FS,
        length,
        ParameterRanges(
            ranges.tc_min, ranges.tc_max, ranges.tc_step,
            ranges.fc_min, ranges.fc_max, ranges.fc_step,
            ranges.logDt_min, ranges.logDt_max, ranges.logDt_step,
            ranges.c_min, ranges.c_max, ranges.c_step
        ),
        verbose
      ) {}


template <typename Scalar>
std::pair<int, Scalar> ACT_MLX_T<Scalar>::search_dictionary(const Eigen::Ref<const act::VecX<Scalar>>& signal) const {
    if (this->get_dict_size() == 0) return {0, Scalar(0)};

#ifdef USE_MLX
    // Only enable MLX fast path for float precision
    if constexpr (std::is_same_v<Scalar, float>) {
        // Lazily ensure MLX dictionary is available
        ensure_mlx_dict();

        const int m = this->get_length();
        const int n = this->get_dict_size();

        // Upload signal as float32 1D array directly from Eigen data (one host->device copy)
        mx::array x_arr(const_cast<float*>(signal.data()), mx::Shape{m}, mx::float32);

        // scores = A^T x, where A is (m x n) row-major on device
        // Compute via matmul(transpose(A), x)
        auto scores = mx::matmul(mx::transpose(*dict_gpu_), x_arr); // shape {n}

        // Argmax and best value
        auto idx_arr = mx::argmax(scores);
        int best_idx = idx_arr.template item<int>();
        auto best_val_arr = mx::take(scores, best_idx);
        float best_val_f = best_val_arr.template item<float>();
        return {best_idx, static_cast<Scalar>(best_val_f)};
    }
#endif

    // Fallback: CPU (Accelerate) path
    return Base::search_dictionary(signal);
}

// MLX helpers (only compiled when USE_MLX is enabled)
#ifdef USE_MLX
template <typename Scalar>
void ACT_MLX_T<Scalar>::ensure_mlx_dict() const {
    // Only meaningful for float specialization
    if constexpr (!std::is_same_v<Scalar, float>) {
        return;
    }
    const int m = this->get_length();
    const int n = this->get_dict_size();

    bool need_pack = !mlx_ready_ || !dict_gpu_ || dict_gpu_->shape(0) != m || dict_gpu_->shape(1) != n;
    if (!need_pack) return;

    // Pack Eigen column-major dict_mat (m x n) into row-major float buffer [m * n]
    dict_rowmajor_.assign(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
    const auto& A = this->get_dict_mat();
    for (int i = 0; i < m; ++i) {
        float* row_ptr = dict_rowmajor_.data() + static_cast<size_t>(i) * static_cast<size_t>(n);
        for (int j = 0; j < n; ++j) {
            row_ptr[j] = static_cast<float>(A(i, j));
        }
    }

    // Upload to MLX device array (row-major)
    dict_gpu_.reset(new mx::array(dict_rowmajor_.data(), mx::Shape{m, n}, mx::float32));
    mlx_ready_ = true;
}
#endif

template <typename Scalar>
int ACT_MLX_T<Scalar>::generate_chirplet_dictionary() {
    // Build using the Accelerate/Eigen base implementation
    int n = Base::generate_chirplet_dictionary();

#ifdef USE_MLX
    // For float32, pre-pack dictionary and warm up kernels so first search is fast
    if constexpr (std::is_same_v<Scalar, float>) {
        // Ensure device dictionary is ready
        ensure_mlx_dict();

        // Warm up: run a dummy matmul/argmax to trigger compilation + placement
        const int m = this->get_length();
        auto x0 = mx::zeros(mx::Shape{m}); // float32 by default
        auto scores = mx::matmul(mx::transpose(*dict_gpu_), x0);
        auto idx = mx::argmax(scores);
        // Force evaluation to complete warmup
        (void)idx.template item<int>();
    }
#endif
    return n;
}

template <typename Scalar>
void ACT_MLX_T<Scalar>::on_dictionary_loaded() {
#ifdef USE_MLX
    // For float32, pre-pack dictionary and warm up kernels so first search is fast
    if constexpr (std::is_same_v<Scalar, float>) {
        // Ensure device dictionary is ready
        ensure_mlx_dict();

        // Warm up: run a dummy matmul/argmax to trigger compilation + placement
        const int m = this->get_length();
        auto x0 = mx::zeros(mx::Shape{m}); // float32 by default
        auto scores = mx::matmul(mx::transpose(*dict_gpu_), x0);
        auto idx = mx::argmax(scores);
        // Force evaluation to complete warmup
        (void)idx.template item<int>();
    }
#endif
}

// Explicit instantiation for double (default alias) and float
template class ACT_MLX_T<double>;
template class ACT_MLX_T<float>;
