#include "ACT_CPU.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

static std::vector<float> generate_test_signal_f(int length, double fs) {
    std::vector<float> signal(length, 0.0f);
    std::vector<float> t(length);
    for (int i = 0; i < length; ++i) t[i] = static_cast<float>(static_cast<double>(i) / fs);

    // Chirplet 1
    for (int i = 0; i < length; ++i) {
        float tc = 0.3f;
        float fc = 5.0f;
        float c  = 10.0f;
        float dt = 0.1f;
        float time_diff = t[i] - tc;
        float gaussian = std::exp(-0.5f * std::pow(time_diff / dt, 2.0f));
        float phase = static_cast<float>(2.0 * M_PI) * (c * time_diff * time_diff + fc * time_diff);
        signal[i] += 0.8f * gaussian * std::cos(phase);
    }
    // Chirplet 2
    for (int i = 0; i < length; ++i) {
        float tc = 0.6f;
        float fc = 8.0f;
        float c  = -5.0f;
        float dt = 0.15f;
        float time_diff = t[i] - tc;
        float gaussian = std::exp(-0.5f * std::pow(time_diff / dt, 2.0f));
        float phase = static_cast<float>(2.0 * M_PI) * (c * time_diff * time_diff + fc * time_diff);
        signal[i] += 0.6f * gaussian * std::cos(phase);
    }
    return signal;
}

int main() {
    std::cout << "=== ACT_CPU_f Test (Eigen + CBLAS, float32) ===\n\n";
    try {
        const double fs = 64.0;
        const int length = 32;

        ACT_CPU_f::ParameterRanges ranges(
            0, 32, 8,
            2, 12, 2,
            -3, -1, 1,
            -10, 10, 10
        );

        ACT_CPU_f act(fs, length, ranges, true);
        int dict_size = act.generate_chirplet_dictionary();
        std::cout << "Dictionary size: " << dict_size << "\n";

        auto signal_vec = generate_test_signal_f(length, fs);

        auto search = act.search_dictionary(signal_vec);
        std::cout << "Best idx: " << search.first << ", value: " << std::setprecision(6) << static_cast<double>(search.second) << "\n";

        auto result = act.transform(signal_vec, 3);
        std::cout << "Final error: " << std::setprecision(6) << static_cast<double>(result.error) << "\n";

        std::cout << "Params (top rows):\n";
        for (int i = 0; i < result.params.rows(); ++i) {
            std::cout << "  [" << i << "] tc=" << static_cast<double>(result.params(i,0))
                      << ", fc=" << static_cast<double>(result.params(i,1))
                      << ", logDt=" << static_cast<double>(result.params(i,2))
                      << ", c=" << static_cast<double>(result.params(i,3))
                      << ", a=" << static_cast<double>(result.coeffs[i]) << "\n";
        }

        std::cout << "\n=== ACT_CPU_f Test complete ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
