#ifndef ACT_CPU_MT_H
#define ACT_CPU_MT_H

#include "ACT_CPU.h"
#include <future>
#include <thread>
#include <vector>

// Header-only utilities to run ACT_CPU-compatible transforms in parallel across batches
// Works with ACT_CPU and subclasses like ACT_Accelerate (methods are const)
namespace actmt {

inline int default_threads(int requested = 0) {
    if (requested > 0) return requested;
    unsigned int hc = std::thread::hardware_concurrency();
    return hc ? static_cast<int>(hc) : 4;
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
