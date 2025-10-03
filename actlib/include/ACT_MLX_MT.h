#ifndef ACT_MLX_MT_H
#define ACT_MLX_MT_H

#include "ACT_MLX.h"
#include "ACT_CPU_MT.h" // for CPU GEMM fallback paths
#include <vector>
#include <type_traits>

#ifdef USE_MLX
#include "mlx/mlx.h"
namespace mx = mlx::core;
#endif

// Header-only batched transforms for the MLX backend (GPU coarse, optional CPU refine)
// No thread-parallel variants here. When MLX is unavailable, fallback to CPU GEMM paths.
namespace actmlx {

// Helpers to deduce scalar from Eigen matrix type
template <typename T>
using eigen_scalar_t = typename std::decay_t<T>::Scalar;

// Batched coarse-only transform using MLX GEMM for coarse selection.
// For float MLX backend: compute S = A^T * R on device, argmax |S| along axis=0.
// For double backend or when USE_MLX is off: fallback to actmt::transform_batch_gemm_coarse_only.
template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch_mlx_gemm_coarse_only(const TAct& act,
                                     const std::vector<Eigen::VectorXd>& signals,
                                     const ACT_CPU::TransformOptions& opts) {
    using ResultT = typename TAct::TransformResult;
    const int m = act.get_length();
    const int n = act.get_dict_size();
    const int k = static_cast<int>(signals.size());
    if (k == 0) return {};

#ifndef USE_MLX
    // Fallback: CPU GEMM path
    return actmt::transform_batch_gemm_coarse_only(act, signals, opts);
#else
    using DictScalar = eigen_scalar_t<decltype(act.get_dict_mat())>;
    if constexpr (!std::is_same_v<DictScalar, float>) {
        // Only float specialization uses MLX for coarse search
        return actmt::transform_batch_gemm_coarse_only(act, signals, opts);
    } else {
        // Float32 MLX path
        const auto& A = act.get_dict_mat();
        const auto& P = act.get_param_mat();
        const mx::array& A_gpu = act.get_dict_gpu(); // {m, n} float32

        // Residuals (m x k), results
        std::vector<ResultT> results(k);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> R(m, k);
        std::vector<int> used_orders(k, 0);
        std::vector<char> active(k, 1);
        std::vector<float> prev_rn2(k, 0.0f);
        int active_count = k;

        for (int j = 0; j < k; ++j) {
            if (signals[j].size() != m) throw std::runtime_error("Signal length mismatch in batch");
            // Cast input double signal to float residual
            R.col(j) = signals[j].cast<float>();
            results[j].params = act::ParamsMat<float>(opts.order, 4);
            results[j].coeffs = act::VecX<float>::Zero(opts.order);
            results[j].signal = R.col(j);
            results[j].approx = act::VecX<float>::Zero(m);
            results[j].residue = R.col(j);
            prev_rn2[j] = act::blas::dot(m, results[j].residue.data(), 1, results[j].residue.data(), 1);
        }

        // Scratch buffer for uploading R to device (row-major)
        std::vector<float> R_rowmajor(static_cast<size_t>(m) * static_cast<size_t>(k));

        for (int it = 0; it < opts.order; ++it) {
            if (active_count == 0) break;

            // Upload R (m x k) as row-major float32
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                    R_rowmajor[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)] = R(i, j);
                }
            }
            mx::array R_gpu(R_rowmajor.data(), mx::Shape{m, k});

            // S = A^T * R  => {n, k}
            auto S = mx::matmul(mx::transpose(A_gpu), R_gpu);
            auto S_abs = mx::abs(S);
            // Argmax by magnitude along rows (axis=0): indices shape {k}
            auto idxs = mx::argmax(S_abs, 0);

            for (int j = 0; j < k; ++j) {
                if (!active[j]) continue;
                // Fetch best index for column j
                int best_idx = mx::take(idxs, j).template item<int>();

                // Coarse params
                Eigen::Vector4d init;
                init[0] = P(best_idx, 0);
                init[1] = P(best_idx, 1);
                init[2] = P(best_idx, 2);
                init[3] = P(best_idx, 3);

                // Atom from dictionary column (float)
                const float* atom = A.col(best_idx).data();
                // Coefficient (host) against current float residual
                float coeff = act::blas::dot(m, atom, 1, R.col(j).data(), 1);

                // Store
                results[j].params.row(it) << static_cast<float>(init[0]), static_cast<float>(init[1]), static_cast<float>(init[2]), static_cast<float>(init[3]);
                results[j].coeffs[it] = coeff;

                // Update residue and approx
                act::blas::axpy(m, -coeff, atom, 1, R.col(j).data(), 1);
                act::blas::axpy(m,  coeff, atom, 1, results[j].approx.data(), 1);

                // Early stopping per-signal
                float pre = prev_rn2[j];
                float post = act::blas::dot(m, R.col(j).data(), 1, R.col(j).data(), 1);
                float delta = pre - post;
                prev_rn2[j] = post;
                if (static_cast<double>(delta) < opts.residual_threshold) {
                    used_orders[j] = it + 1;
                    active[j] = 0;
                    --active_count;
                }
            }
        }

        // Finalize
        for (int j = 0; j < k; ++j) {
            results[j].residue = R.col(j);
            float rn2 = act::blas::dot(m, results[j].residue.data(), 1, results[j].residue.data(), 1);
            results[j].error = static_cast<float>(std::sqrt(static_cast<double>(rn2)));
            int uo = used_orders[j] ? used_orders[j] : opts.order;
            if (uo < opts.order) {
                results[j].params.conservativeResize(uo, 4);
                results[j].coeffs.conservativeResize(uo);
            }
        }

        return results;
    }
#endif
}

// Refine-enabled batched transform: MLX coarse selection + per-signal CPU BFGS refinement (on the current residual)
template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch_mlx_gemm(const TAct& act,
                         const std::vector<Eigen::VectorXd>& signals,
                         const ACT_CPU::TransformOptions& opts) {
    using ResultT = typename TAct::TransformResult;
    const int k = static_cast<int>(signals.size());
    if (k == 0) return {};
    const int m = act.get_length();
    const int n = act.get_dict_size();

#ifndef USE_MLX
    return actmt::transform_batch_gemm(act, signals, opts);
#else
    using DictScalar = eigen_scalar_t<decltype(act.get_dict_mat())>;
    if constexpr (!std::is_same_v<DictScalar, float>) {
        return actmt::transform_batch_gemm(act, signals, opts);
    } else {
        const auto& A = act.get_dict_mat();
        const auto& P = act.get_param_mat();
        const mx::array& A_gpu = act.get_dict_gpu();

        std::vector<ResultT> results(k);
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> R(m, k);
        std::vector<int> used_orders(k, 0);
        std::vector<char> active(k, 1);
        std::vector<float> prev_rn2(k, 0.0f);
        int active_count = k;

        for (int j = 0; j < k; ++j) {
            if (signals[j].size() != m) throw std::runtime_error("Signal length mismatch in batch");
            R.col(j) = signals[j].cast<float>();
            results[j].params = act::ParamsMat<float>(opts.order, 4);
            results[j].coeffs = act::VecX<float>::Zero(opts.order);
            results[j].signal = R.col(j);
            results[j].approx = act::VecX<float>::Zero(m);
            results[j].residue = R.col(j);
            prev_rn2[j] = act::blas::dot(m, results[j].residue.data(), 1, results[j].residue.data(), 1);
        }

        std::vector<float> R_rowmajor(static_cast<size_t>(m) * static_cast<size_t>(k));

        for (int it = 0; it < opts.order; ++it) {
            if (active_count == 0) break;

            // Upload R
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                    R_rowmajor[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)] = R(i, j);
                }
            }
            mx::array R_gpu(R_rowmajor.data(), mx::Shape{m, k});

            // Coarse selection on device
            auto S = mx::matmul(mx::transpose(A_gpu), R_gpu); // {n, k}
            auto idxs = mx::argmax(mx::abs(S), 0);            // {k}

            for (int j = 0; j < k; ++j) {
                if (!active[j]) continue;
                int best_idx = mx::take(idxs, j).template item<int>();

                // Coarse init
                Eigen::Vector4d init;
                init[0] = P(best_idx, 0);
                init[1] = P(best_idx, 1);
                init[2] = P(best_idx, 2);
                init[3] = P(best_idx, 3);

                // Refine in double (public wrapper handles internal double optimize)
                Eigen::Vector4d refined = init;
                if (opts.refine) {
                    // Map current float residual to Eigen::Map<float> and pass; wrapper accepts Scalar=Scalar(act)
                    refined = act.refine_params_bfgs(init, R.col(j));
                }

                // Generate atom from refined params (float)
                auto atom = act.g(refined[0], refined[1], refined[2], refined[3]);
                // Coefficient
                float coeff = act::blas::dot(m, atom.data(), 1, R.col(j).data(), 1);

                // Store
                results[j].params.row(it) << static_cast<float>(refined[0]), static_cast<float>(refined[1]), static_cast<float>(refined[2]), static_cast<float>(refined[3]);
                results[j].coeffs[it] = coeff;

                // Update
                act::blas::axpy(m, -coeff, atom.data(), 1, R.col(j).data(), 1);
                act::blas::axpy(m,  coeff, atom.data(), 1, results[j].approx.data(), 1);

                // Early stop
                float pre = prev_rn2[j];
                float post = act::blas::dot(m, R.col(j).data(), 1, R.col(j).data(), 1);
                float delta = pre - post;
                prev_rn2[j] = post;
                if (static_cast<double>(delta) < opts.residual_threshold) {
                    used_orders[j] = it + 1;
                    active[j] = 0;
                    --active_count;
                }
            }
        }

        for (int j = 0; j < k; ++j) {
            results[j].residue = R.col(j);
            float rn2 = act::blas::dot(m, results[j].residue.data(), 1, results[j].residue.data(), 1);
            results[j].error = static_cast<float>(std::sqrt(static_cast<double>(rn2)));
            int uo = used_orders[j] ? used_orders[j] : opts.order;
            if (uo < opts.order) {
                results[j].params.conservativeResize(uo, 4);
                results[j].coeffs.conservativeResize(uo);
            }
        }

        return results;
    }
#endif
}

// Overloads for std::vector<double>
template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch_mlx_gemm_coarse_only(const TAct& act,
                                     const std::vector<std::vector<double>>& signals,
                                     const ACT_CPU::TransformOptions& opts) {
    std::vector<Eigen::VectorXd> xs;
    xs.reserve(signals.size());
    for (const auto& v : signals) {
        xs.emplace_back(Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<int>(v.size())));
    }
    return transform_batch_mlx_gemm_coarse_only(act, xs, opts);
}

template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch_mlx_gemm(const TAct& act,
                         const std::vector<std::vector<double>>& signals,
                         const ACT_CPU::TransformOptions& opts) {
    std::vector<Eigen::VectorXd> xs;
    xs.reserve(signals.size());
    for (const auto& v : signals) {
        xs.emplace_back(Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<int>(v.size())));
    }
    return transform_batch_mlx_gemm(act, xs, opts);
}

// Dispatcher: choose MLX batched GEMM coarse-only when refine==false, else MLX batched GEMM + refine.
// Falls back to CPU GEMM if MLX is unavailable or backend is double.
template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch(const TAct& act,
                const std::vector<Eigen::VectorXd>& signals,
                const ACT_CPU::TransformOptions& opts) {
    if (!opts.refine) return transform_batch_mlx_gemm_coarse_only(act, signals, opts);
    return transform_batch_mlx_gemm(act, signals, opts);
}

template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch(const TAct& act,
                const std::vector<std::vector<double>>& signals,
                const ACT_CPU::TransformOptions& opts) {
    if (!opts.refine) return transform_batch_mlx_gemm_coarse_only(act, signals, opts);
    return transform_batch_mlx_gemm(act, signals, opts);
}

} // namespace actmlx

#endif // ACT_MLX_MT_H
