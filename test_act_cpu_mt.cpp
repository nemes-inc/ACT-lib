#include "ACT_CPU.h"
#include "ACT_Accelerate.h"
#include "ACT_CPU_MT.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

static ACT_CPU::ParameterRanges make_ranges(int length) {
    return ACT_CPU::ParameterRanges(
        0, length - 1, std::max(1.0, std::floor(length / 64.0)),
        2.0, 20.0, 0.5,
        -3.0, -1.0, 0.5,
        -10.0, 10.0, 2.0
    );
}

static std::vector<std::vector<double>> make_batch(int length, int batch, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> N(0.0, 1.0);
    std::vector<std::vector<double>> xs(batch, std::vector<double>(length));
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < length; ++i) xs[b][i] = N(rng);
        // zero-mean
        double mean = 0.0; for (double v : xs[b]) mean += v; mean /= length;
        for (double& v : xs[b]) v -= mean;
    }
    return xs;
}

static bool is_close_vec(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol = 1e-9) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    try {
        const double fs = 128.0;
        const int length = 256;
        const int batch = 6;

        auto ranges = make_ranges(length);

        // 1) ACT_CPU single-instance shared across threads
        ACT_CPU act_cpu(fs, length, ranges, false);
        act_cpu.generate_chirplet_dictionary();

        auto signals = make_batch(length, batch, 123);

        ACT_CPU::TransformOptions opts; opts.order = 3; opts.refine = false; opts.residual_threshold = 1e-6;

        // Serial
        std::vector<ACT_CPU::TransformResult> serial;
        serial.reserve(batch);
        for (const auto& s : signals) {
            Eigen::Map<const Eigen::VectorXd> x(s.data(), length);
            serial.emplace_back(act_cpu.transform(x, opts));
        }

        // Parallel
        auto parallel = actmt::transform_batch_parallel(act_cpu, signals, opts, 0);

        // Compare
        for (int i = 0; i < batch; ++i) {
            if (!is_close_vec(serial[i].approx, parallel[i].approx, 1e-9)) {
                std::cerr << "Mismatch in approx for item " << i << std::endl;
                return 1;
            }
            if (!is_close_vec(serial[i].residue, parallel[i].residue, 1e-9)) {
                std::cerr << "Mismatch in residue for item " << i << std::endl;
                return 1;
            }
        }
        std::cout << "ACT_CPU MT test (coarse-only) passed.\n";

        // 2) ACT_Accelerate path (falls back to ACT_CPU if not on Apple)
        ACT_Accelerate act_acc(fs, length, ranges, false);
        act_acc.generate_chirplet_dictionary();

        // Serial
        std::vector<ACT_CPU::TransformResult> serial_acc;
        serial_acc.reserve(batch);
        for (const auto& s : signals) {
            Eigen::Map<const Eigen::VectorXd> x(s.data(), length);
            serial_acc.emplace_back(act_acc.transform(x, opts));
        }
        // Parallel
        auto parallel_acc = actmt::transform_batch_parallel(act_acc, signals, opts, 0);
        for (int i = 0; i < batch; ++i) {
            if (!is_close_vec(serial_acc[i].approx, parallel_acc[i].approx, 1e-9)) {
                std::cerr << "ACC mismatch in approx for item " << i << std::endl;
                return 1;
            }
            if (!is_close_vec(serial_acc[i].residue, parallel_acc[i].residue, 1e-9)) {
                std::cerr << "ACC mismatch in residue for item " << i << std::endl;
                return 1;
            }
        }
        std::cout << "ACT_Accelerate MT test (coarse-only) passed.\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
