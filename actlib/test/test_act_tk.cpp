#include "ACT_CUDA_TK.h"
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
        float tc = 0.30f;
        float fc = 5.0f;
        float c  = 10.0f;
        float dt = 0.10f;
        float time_diff = t[i] - tc;
        float gaussian = std::exp(-0.5f * std::pow(time_diff / dt, 2.0f));
        float phase = static_cast<float>(2.0 * M_PI) * (c * time_diff * time_diff + fc * time_diff);
        signal[i] += 0.8f * gaussian * std::cos(phase);
    }
    // Chirplet 2
    for (int i = 0; i < length; ++i) {
        float tc = 0.65f;
        float fc = 8.0f;
        float c  = -6.0f;
        float dt = 0.15f;
        float time_diff = t[i] - tc;
        float gaussian = std::exp(-0.5f * std::pow(time_diff / dt, 2.0f));
        float phase = static_cast<float>(2.0 * M_PI) * (c * time_diff * time_diff + fc * time_diff);
        signal[i] += 0.6f * gaussian * std::cos(phase);
    }
    return signal;
}

int main() {
    std::cout << "=== ACT_CUDA_TK_f Smoke Test (custom CUDA GEMV + host argmax) ===\n\n";
    try {
        const double fs = 128.0;
        const int length = 128;

        // Modest grid to keep dictionary small
        ACT_CUDA_TK_f::ParameterRanges ranges(
            0, length, 16,    // tc
            2.0, 12.0, 2.0,   // fc
            -3.0, -1.0, 1.0,  // logDt
            -10.0, 10.0, 10.0 // c
        );

        ACT_CUDA_TK_f act(fs, length, ranges, true);
        int dict_size = act.generate_chirplet_dictionary();
        std::cout << "Dictionary generated: " << dict_size << " atoms\n";

        auto sig = generate_test_signal_f(length, fs);

        // Quick dictionary search (custom kernel path for float32)
        auto best = act.search_dictionary(sig);
        std::cout << "Best idx: " << best.first << ", value: " << std::setprecision(6) << best.second << "\n";

        // Small transform to exercise repeated searches and updates
        auto result = act.transform(sig, 2);
        std::cout << "Transform complete. Error=" << std::fixed << std::setprecision(6) << result.error << "\n";
        std::cout << "Chirplets found: " << result.params.rows() << "\n";
        for (int i = 0; i < result.params.rows(); ++i) {
            std::cout << "  #" << (i+1) << ": tc=" << result.params(i,0)
                      << ", fc=" << result.params(i,1)
                      << ", logDt=" << result.params(i,2)
                      << ", c=" << result.params(i,3)
                      << ", coeff=" << result.coeffs[i] << "\n";
        }

        std::cout << "\n=== ACT_CUDA_TK_f Smoke Test complete ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
