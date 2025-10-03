#include "ACT_CPU.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

static std::vector<double> generate_test_signal(int length, double fs) {
    std::vector<double> signal(length, 0.0);
    std::vector<double> t(length);
    for (int i = 0; i < length; ++i) t[i] = static_cast<double>(i) / fs;

    // Chirplet 1
    for (int i = 0; i < length; ++i) {
        double tc = 0.3;
        double fc = 5.0;
        double c = 10.0;
        double dt = 0.1;
        double time_diff = t[i] - tc;
        double gaussian = std::exp(-0.5 * std::pow(time_diff / dt, 2));
        double phase = 2.0 * M_PI * (c * time_diff * time_diff + fc * time_diff);
        signal[i] += 0.8 * gaussian * std::cos(phase);
    }
    // Chirplet 2
    for (int i = 0; i < length; ++i) {
        double tc = 0.6;
        double fc = 8.0;
        double c = -5.0;
        double dt = 0.15;
        double time_diff = t[i] - tc;
        double gaussian = std::exp(-0.5 * std::pow(time_diff / dt, 2));
        double phase = 2.0 * M_PI * (c * time_diff * time_diff + fc * time_diff);
        signal[i] += 0.6 * gaussian * std::cos(phase);
    }
    return signal;
}

int main() {
    std::cout << "=== ACT_CPU Test (Eigen + CBLAS) ===\n\n";
    try {
        const double fs = 64.0;
        const int length = 32;

        ACT_CPU::ParameterRanges ranges(
            0, 32, 8,
            2, 12, 2,
            -3, -1, 1,
            -10, 10, 10
        );

        ACT_CPU act(fs, length, ranges, true);
        int dict_size = act.generate_chirplet_dictionary();
        std::cout << "Dictionary size: " << dict_size << "\n";

        auto signal_vec = generate_test_signal(length, fs);
        Eigen::Map<const Eigen::VectorXd> signal(signal_vec.data(), length);

        auto search = act.search_dictionary(signal);
        std::cout << "Best idx: " << search.first << ", value: " << std::setprecision(6) << search.second << "\n";

        auto result = act.transform(signal, 3);
        std::cout << "Final error: " << std::setprecision(6) << result.error << "\n";

        std::cout << "Params (top rows):\n";
        for (int i = 0; i < result.params.rows(); ++i) {
            std::cout << "  [" << i << "] tc=" << result.params(i,0)
                      << ", fc=" << result.params(i,1)
                      << ", logDt=" << result.params(i,2)
                      << ", c=" << result.params(i,3)
                      << ", a=" << result.coeffs[i] << "\n";
        }

        // Save/Load sanity test
        const std::string path = "act_cpu_test_dict.bin";
        std::cout << "Attempting to save dictionary to '" << path << "'...\n";
        if (act.save_dictionary(path)) {
            std::cout << "Saved. Attempting to load...\n";
            auto loaded = ACT_CPU::load_dictionary<ACT_CPU>(path, false);
            if (loaded) {
                std::cout << "Reloaded dict_size=" << loaded->get_dict_size() << ", length=" << loaded->get_length() << "\n";
            } else {
                std::cout << "Failed to reload dictionary." << std::endl;
            }
        } else {
            std::cout << "Failed to save dictionary." << std::endl;
        }

        std::cout << "\n=== ACT_CPU Test complete ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
