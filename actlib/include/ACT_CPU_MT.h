#ifndef ACT_CPU_MT_H
#define ACT_CPU_MT_H

#include "ACT_CPU.h"
#include "act_numeric.h"
#include <future>
#include <thread>
#include <vector>
#include <limits>
#include <cmath>

// Header-only utilities to run ACT_CPU-compatible transforms in parallel across batches
// Works with ACT_CPU and subclasses like ACT_Accelerate (methods are const)
namespace actmt {

// Helper: deduce Eigen matrix scalar type
template <typename T>
using eigen_scalar_t = typename std::decay_t<T>::Scalar;

inline int default_threads(int requested = 0) {
    if (requested > 0) return requested;
    unsigned int hc = std::thread::hardware_concurrency();
    return hc ? static_cast<int>(hc) : 4;
}

// Batched transform using GEMM for coarse selection, then optional BFGS refinement per signal
// Mirrors ACT_CPU::transform semantics across a batch, sharing the coarse GEMM.
template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch_gemm(const TAct& act,
                     const std::vector<Eigen::VectorXd>& signals,
                     const ACT_CPU::TransformOptions& opts) {
    using ResultT = typename TAct::TransformResult;
    using Scalar = eigen_scalar_t<decltype(act.get_dict_mat())>;
    const int m = act.get_length();
    const int n = act.get_dict_size();
    const int k = static_cast<int>(signals.size());
    if (k == 0) return {};

    // Pack residuals R (m x k)
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> R(m, k);
    std::vector<ResultT> results(k);
    std::vector<int> used_orders(k, 0);
    std::vector<char> active(k, 1);
    std::vector<Scalar> prev_rn2(k, Scalar(0));
    int active_count = k;

    for (int j = 0; j < k; ++j) {
        if (signals[j].size() != m) throw std::runtime_error("Signal length mismatch in batch");
        // Cast input double signal to backend scalar type
        R.col(j) = signals[j].template cast<Scalar>();
        results[j].params = act::ParamsMat<Scalar>(opts.order, 4);
        results[j].coeffs = act::VecX<Scalar>::Zero(opts.order);
        results[j].signal = R.col(j);
        results[j].approx = act::VecX<Scalar>::Zero(m);
        results[j].residue = R.col(j);
        prev_rn2[j] = act::blas::dot(m, results[j].residue.data(), 1, results[j].residue.data(), 1);
    }

    const auto& A = act.get_dict_mat();
    const auto& P = act.get_param_mat();

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> S(n, k);

    for (int it = 0; it < opts.order; ++it) {
        if (active_count == 0) break;
        // Coarse scores for all signals at once: S = A^T * R
        act::blas::gemm_colmajor(CblasTrans, CblasNoTrans,
                                  n, k, m,
                                  Scalar(1),
                                  A.data(), m,
                                  R.data(), m,
                                  Scalar(0),
                                  S.data(), n);

        for (int j = 0; j < k; ++j) {
            if (!active[j]) continue;
            const Scalar* Sj = S.data() + static_cast<size_t>(j) * static_cast<size_t>(n);
            int best_idx = act::blas::iamax(n, Sj, 1);
            Scalar coarse_coeff = Sj[best_idx];
            (void)coarse_coeff; // not used directly, coeff recomputed below
            // Coarse init params from dictionary grid (double grid)
            Eigen::Vector4d init;
            init[0] = P(best_idx, 0);
            init[1] = P(best_idx, 1);
            init[2] = P(best_idx, 2);
            init[3] = P(best_idx, 3);

            // Refinement if requested
            Eigen::Vector4d refined = init;
            if (opts.refine) {
                refined = act.refine_params_bfgs(init, R.col(j));
            }

            // Atom and coefficient with current residue
            auto atom = act.g(refined[0], refined[1], refined[2], refined[3]);
            Scalar coeff = act::blas::dot(m, atom.data(), 1, R.col(j).data(), 1);

            // Store
            results[j].params.row(it) << static_cast<Scalar>(refined[0]), static_cast<Scalar>(refined[1]), static_cast<Scalar>(refined[2]), static_cast<Scalar>(refined[3]);
            results[j].coeffs[it] = coeff;

            // Update residue and approx
            act::blas::axpy(m, Scalar(-1) * coeff, atom.data(), 1, R.col(j).data(), 1);
            act::blas::axpy(m, coeff, atom.data(), 1, results[j].approx.data(), 1);

            // Early stopping per-signal
            Scalar pre = prev_rn2[j];
            Scalar post = act::blas::dot(m, R.col(j).data(), 1, R.col(j).data(), 1);
            double delta = static_cast<double>(pre) - static_cast<double>(post);
            prev_rn2[j] = post;
            if (delta < opts.residual_threshold) {
                used_orders[j] = it + 1;
                active[j] = 0;
                --active_count;
            }
        }
    }

    // Finalize
    for (int j = 0; j < k; ++j) {
        results[j].residue = R.col(j);
        Scalar rn2 = act::blas::dot(m, results[j].residue.data(), 1, results[j].residue.data(), 1);
        results[j].error = static_cast<Scalar>(std::sqrt(static_cast<double>(rn2)));
        int uo = used_orders[j] ? used_orders[j] : opts.order;
        if (uo < opts.order) {
            results[j].params.conservativeResize(uo, 4);
            results[j].coeffs.conservativeResize(uo);
        }
    }

    return results;
}

// Overload for std::vector<double>
template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch_gemm(const TAct& act,
                     const std::vector<std::vector<double>>& signals,
                     const ACT_CPU::TransformOptions& opts) {
    std::vector<Eigen::VectorXd> xs;
    xs.reserve(signals.size());
    for (const auto& v : signals) xs.emplace_back(Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<int>(v.size())));
    return transform_batch_gemm(act, xs, opts);
}

// Batched coarse-only transform using a single BLAS GEMM per iteration
// Computes Scores = A^T * R, where A is the dictionary (m x n), R is residuals (m x k)
// Then selects per-signal argmax by magnitude and updates residues/approx.
// Notes:
// - Uses max |score| (magnitude) to match energy drop criterion (coeff^2)
// - Early stops per signal when coeff^2 < residual_threshold
// - Returns the same structure as ACT_CPU::transform with refine=false
template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch_gemm_coarse_only(const TAct& act,
                                 const std::vector<Eigen::VectorXd>& signals,
                                 const ACT_CPU::TransformOptions& opts) {
    using ResultT = typename TAct::TransformResult;
    using Scalar = eigen_scalar_t<decltype(act.get_dict_mat())>;
    const int m = act.get_length();
    const int n = act.get_dict_size();
    const int k = static_cast<int>(signals.size());
    if (k == 0) return {};

    // Pack residuals R (m x k) column-major
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> R(m, k);
    std::vector<ResultT> results(k);
    std::vector<int> used_orders(k, 0);
    std::vector<char> active(k, 1);
    std::vector<Scalar> prev_rn2(k, Scalar(0));
    int active_count = k;

    for (int j = 0; j < k; ++j) {
        if (signals[j].size() != m) throw std::runtime_error("Signal length mismatch in batch");
        R.col(j) = signals[j].template cast<Scalar>();
        results[j].params = act::ParamsMat<Scalar>(opts.order, 4);
        results[j].coeffs = act::VecX<Scalar>::Zero(opts.order);
        results[j].signal = R.col(j);
        results[j].approx = act::VecX<Scalar>::Zero(m);
        results[j].residue = R.col(j);
        // Initial residual norm^2 per signal (matches ACT_CPU::transform)
        prev_rn2[j] = act::blas::dot(m, results[j].residue.data(), 1, results[j].residue.data(), 1);
    }

    const auto& A = act.get_dict_mat();              // (m x n) column-major
    const auto& P = act.get_param_mat();             // (n x 4) row-major (double)

    // Scores (n x k)
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> S(n, k);

    for (int it = 0; it < opts.order; ++it) {
        if (active_count == 0) break;
        // S = A^T * R  => (n x k) = (n x m) * (m x k)
        act::blas::gemm_colmajor(CblasTrans, CblasNoTrans,
                                  n, k, m,
                                  Scalar(1),
                                  A.data(), m,
                                  R.data(), m,
                                  Scalar(0),
                                  S.data(), n);

        for (int j = 0; j < k; ++j) {
            if (!active[j]) continue;
            // Column j of S is contiguous of length n, stride 1
            int best_idx = act::blas::iamax(n, S.data() + static_cast<size_t>(j) * static_cast<size_t>(n), 1);
            Scalar coeff = *(S.data() + static_cast<size_t>(j) * static_cast<size_t>(n) + best_idx);

            // Store params and coeff
            results[j].params.row(it) << static_cast<Scalar>(P(best_idx, 0)), static_cast<Scalar>(P(best_idx, 1)), static_cast<Scalar>(P(best_idx, 2)), static_cast<Scalar>(P(best_idx, 3));
            results[j].coeffs[it] = coeff;

            // Update residue R[:,j] and approx
            const Scalar* atom = A.col(best_idx).data();
            act::blas::axpy(m, Scalar(-1) * coeff, atom, 1, R.col(j).data(), 1);  // R[:,j] -= coeff * atom
            act::blas::axpy(m, coeff, atom, 1, results[j].approx.data(), 1);       // approx_j += coeff * atom

            // Early stop for this signal if delta residual norm^2 < threshold
            Scalar pre = prev_rn2[j];
            Scalar post = act::blas::dot(m, R.col(j).data(), 1, R.col(j).data(), 1);
            double delta = static_cast<double>(pre) - static_cast<double>(post);
            prev_rn2[j] = post;
            if (delta < opts.residual_threshold) {
                used_orders[j] = it + 1;
                active[j] = 0;
                --active_count;
            }
        }
    }

    // Finalize results: set residue, error, shrink to used orders
    for (int j = 0; j < k; ++j) {
        // Residue vector from R
        results[j].residue = R.col(j);
        // Compute error = ||residue||_2
        Scalar rn2 = act::blas::dot(m, results[j].residue.data(), 1, results[j].residue.data(), 1);
        results[j].error = static_cast<Scalar>(std::sqrt(static_cast<double>(rn2)));
        int uo = used_orders[j] ? used_orders[j] : opts.order;
        if (uo < opts.order) {
            results[j].params.conservativeResize(uo, 4);
            results[j].coeffs.conservativeResize(uo);
        }
    }

    return results;
}

// Overload: std::vector<double> batch wrapper
template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch_gemm_coarse_only(const TAct& act,
                                 const std::vector<std::vector<double>>& signals,
                                 const ACT_CPU::TransformOptions& opts) {
    std::vector<Eigen::VectorXd> xs;
    xs.reserve(signals.size());
    for (const auto& v : signals) {
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<int>(v.size()));
        xs.emplace_back(std::move(x));
    }
    return transform_batch_gemm_coarse_only(act, xs, opts);
}

// Convenience API: choose GEMM batched coarse-only when refine==false,
// else run the per-signal transform in parallel threads.
template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch(const TAct& act,
                const std::vector<Eigen::VectorXd>& signals,
                const ACT_CPU::TransformOptions& opts,
                int threads = 0) {
    // Always use GEMM for coarse selection; refine toggles BFGS per signal.
    if (!opts.refine) return transform_batch_gemm_coarse_only(act, signals, opts);
    return transform_batch_gemm(act, signals, opts);
}

template <typename TAct>
inline std::vector<typename TAct::TransformResult>
transform_batch(const TAct& act,
                const std::vector<std::vector<double>>& signals,
                const ACT_CPU::TransformOptions& opts,
                int threads = 0) {
    if (!opts.refine) return transform_batch_gemm_coarse_only(act, signals, opts);
    return transform_batch_gemm(act, signals, opts);
}

// Transform a single Eigen vector
template <typename TAct>
inline ACT_CPU::TransformResult transform_one(const TAct& act,
                                              const Eigen::Ref<const Eigen::VectorXd>& x,
                                              const ACT_CPU::TransformOptions& opts) {
    return act.transform(x, opts);
}

// Transform a batch of Eigen vectors in parallel
// TAct must provide: TransformResult transform(const Eigen::Ref<const Eigen::VectorXd>&, const ACT_CPU::TransformOptions&) const;
// and be safe to call concurrently (ACT_CPU/ACT_Accelerate are const and read-only over dictionary)
template <typename TAct>
inline std::vector<ACT_CPU::TransformResult>
transform_batch_parallel(const TAct& act,
                         const std::vector<Eigen::VectorXd>& signals,
                         const ACT_CPU::TransformOptions& opts,
                         int threads = 0) {
    const int nthreads = default_threads(threads);
    std::vector<std::future<ACT_CPU::TransformResult>> futures;
    futures.reserve(signals.size());

    for (const auto& sig : signals) {
        futures.emplace_back(std::async(std::launch::async, [&act, &sig, &opts]() {
            return transform_one(act, sig, opts);
        }));
    }

    std::vector<ACT_CPU::TransformResult> results;
    results.reserve(signals.size());
    for (auto& f : futures) results.emplace_back(f.get());
    return results;
}

// Transform a batch of std::vector<double> signals in parallel
// Avoids copying by mapping each vector to an Eigen::Map on the fly
template <typename TAct>
inline std::vector<ACT_CPU::TransformResult>
transform_batch_parallel(const TAct& act,
                         const std::vector<std::vector<double>>& signals,
                         const ACT_CPU::TransformOptions& opts,
                         int threads = 0) {
    const int nthreads = default_threads(threads);
    std::vector<std::future<ACT_CPU::TransformResult>> futures;
    futures.reserve(signals.size());

    for (const auto& sigvec : signals) {
        futures.emplace_back(std::async(std::launch::async, [&act, &sigvec, &opts]() {
            Eigen::Map<const Eigen::VectorXd> x(sigvec.data(), static_cast<int>(sigvec.size()));
            return act.transform(x, opts);
        }));
    }

    std::vector<ACT_CPU::TransformResult> results;
    results.reserve(signals.size());
    for (auto& f : futures) results.emplace_back(f.get());
    return results;
}

} // namespace actmt

#endif // ACT_CPU_MT_H
