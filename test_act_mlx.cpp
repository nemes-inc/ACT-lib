#include "ACT_MLX.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Small helper to generate a simple test signal
static std::vector<double> generate_signal(int length, double fs) {
    std::vector<double> s(length, 0.0);
    for (int i = 0; i < length; ++i) {
        double t = static_cast<double>(i) / fs;
        double g1 = std::exp(-0.5 * std::pow((t - 0.30) / 0.10, 2));
        double p1 = 2.0 * M_PI * (10.0 * (t - 0.30) * (t - 0.30) + 5.0 * (t - 0.30));
        s[i] += 0.8 * g1 * std::cos(p1);
        double g2 = std::exp(-0.5 * std::pow((t - 0.65) / 0.15, 2));
        double p2 = 2.0 * M_PI * (-6.0 * (t - 0.65) * (t - 0.65) + 8.0 * (t - 0.65));
        s[i] += 0.6 * g2 * std::cos(p2);
    }
    return s;
}

int main() {
    std::cout << "=== ACT_MLX test (inherits Accelerate path) ===\n";
    try {
        double fs = 128.0;
        int length = 128;

        // Small parameter ranges for quick test
        ACT::ParameterRanges ranges(
            0, length, 16,    // tc
            2.0, 12.0, 2.0,   // fc
            -3.0, -1.0, 1.0,  // logDt
            -10.0, 10.0, 10.0 // c
        );

        ACT_MLX act(fs, length, ranges, true);

        int dict_size = act.generate_chirplet_dictionary();
        std::cout << "Dictionary generated: " << dict_size << " atoms\n";

        auto sig = generate_signal(length, fs);
        auto res = act.transform(sig, 2);

        std::cout << "Transform complete. Error=" << std::fixed << std::setprecision(6) << res.error << "\n";
        std::cout << "Chirplets found: " << res.params.rows() << "\n";
        for (int i = 0; i < res.params.rows(); ++i) {
            std::cout << "  #" << (i+1) << ": tc=" << res.params(i,0)
                      << ", fc=" << res.params(i,1)
                      << ", logDt=" << res.params(i,2)
                      << ", c=" << res.params(i,3)
                      << ", coeff=" << res.coeffs[i] << "\n";
        }

        std::cout << "=== OK ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
