#include "ACT.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <limits>

static std::vector<double> generate_test_signal(int length, double fs) {
    std::vector<double> sig(length, 0.0);
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, 0.02);
    for (int i = 0; i < length; ++i) {
        double t = i / fs;
        sig[i] = 0.7 * std::cos(2.0 * M_PI * 6.0 * t) +
                 0.5 * std::cos(2.0 * M_PI * 10.0 * t) +
                 noise(gen);
    }
    return sig;
}

static bool nearly_equal(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) <= tol * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

int main() {
    std::cout << "==============================\n";
    std::cout << "  DICTIONARY SAVE/LOAD TEST\n";
    std::cout << "==============================\n";

    // Small, fast test configuration
    const double FS = 64.0;
    const int LENGTH = 64;
    ACT::ParameterRanges ranges(
        0, LENGTH - 1, 16,   // tc: 5 values
        2.0, 12.0, 2.0,      // fc: 6 values
        -3.0, -1.0, 1.0,     // logDt: 3 values
        -10.0, 10.0, 10.0    // c: 3 values
    );

    // 1) Create dictionary
    std::cout << "1) Creating reference dictionary..." << std::flush;
    ACT act(FS, LENGTH, ranges, false, false);
    int dict_size = act.generate_chirplet_dictionary();
    std::cout << " done. (size=" << dict_size << ")\n";

    // 2) Save dictionary
    const char* dict_path = "dict_io_test.bin";
    std::cout << "2) Saving dictionary to '" << dict_path << "'..." << std::flush;
    if (!act.save_dictionary(dict_path)) {
        std::cerr << "\nERROR: Failed to save dictionary." << std::endl;
        return 1;
    }
    std::cout << " done.\n";

    // 3) Load dictionary into ACT
    std::cout << "3) Loading dictionary into ACT..." << std::flush;
    auto act_loaded = ACT::load_dictionary<ACT>(dict_path);
    if (!act_loaded) {
        std::cerr << "\nERROR: Failed to load dictionary into ACT." << std::endl;
        return 1;
    }
    std::cout << " done.\n";

    // 4) Verify core parameters
    std::cout << "4) Verifying core parameters..." << std::flush;
    bool ok = true;
    ok &= nearly_equal(act_loaded->get_FS(), FS);
    ok &= (act_loaded->get_length() == LENGTH);

    const auto& pr_ref = act.get_param_ranges();
    const auto& pr_ld  = act_loaded->get_param_ranges();
    ok &= nearly_equal(pr_ref.tc_min, pr_ld.tc_min);
    ok &= nearly_equal(pr_ref.tc_max, pr_ld.tc_max);
    ok &= nearly_equal(pr_ref.tc_step, pr_ld.tc_step);
    ok &= nearly_equal(pr_ref.fc_min, pr_ld.fc_min);
    ok &= nearly_equal(pr_ref.fc_max, pr_ld.fc_max);
    ok &= nearly_equal(pr_ref.fc_step, pr_ld.fc_step);
    ok &= nearly_equal(pr_ref.logDt_min, pr_ld.logDt_min);
    ok &= nearly_equal(pr_ref.logDt_max, pr_ld.logDt_max);
    ok &= nearly_equal(pr_ref.logDt_step, pr_ld.logDt_step);
    ok &= nearly_equal(pr_ref.c_min, pr_ld.c_min);
    ok &= nearly_equal(pr_ref.c_max, pr_ld.c_max);
    ok &= nearly_equal(pr_ref.c_step, pr_ld.c_step);
    ok &= (act_loaded->get_dict_size() == act.get_dict_size());

    if (!ok) {
        std::cerr << "\nERROR: Core parameter mismatch after load." << std::endl;
        return 1;
    }
    std::cout << " ok.\n";

    // 5) Deep content check: compare a few entries of dict_mat and param_mat
    std::cout << "5) Deep content verification..." << std::flush;
    const auto& dict_ref = act.get_dict_mat();
    const auto& dict_ld  = act_loaded->get_dict_mat();
    const auto& pm_ref   = act.get_param_mat();
    const auto& pm_ld    = act_loaded->get_param_mat();

    double max_abs_diff = 0.0;
    for (int i = 0; i < act.get_dict_size(); ++i) {
        for (int j = 0; j < LENGTH; ++j) {
            double d = std::fabs(dict_ref[i][j] - dict_ld[i][j]);
            if (d > max_abs_diff) max_abs_diff = d;
        }
        for (int k = 0; k < 4; ++k) {
            double d = std::fabs(pm_ref[i][k] - pm_ld[i][k]);
            if (d > max_abs_diff) max_abs_diff = d;
        }
    }
    if (max_abs_diff > 1e-12) {
        std::cerr << "\nERROR: Dictionary matrices differ after load. max_abs_diff=" << max_abs_diff << std::endl;
        return 1;
    }
    std::cout << " ok (max_abs_diff=" << max_abs_diff << ").\n";

    // 6) Functional check: dictionary search should match
    std::cout << "6) Functional check: dictionary search..." << std::flush;
    auto test_signal = generate_test_signal(LENGTH, FS);
    auto [idx_ref, val_ref] = act.search_dictionary(test_signal);
    auto [idx_ld,  val_ld ] = act_loaded->search_dictionary(test_signal);

    bool search_ok = (idx_ref == idx_ld) && nearly_equal(val_ref, val_ld, 1e-9);
    if (!search_ok) {
        std::cerr << "\nERROR: Search mismatch: ref(" << idx_ref << "," << val_ref
                  << ") vs act_loaded(" << idx_ld << "," << val_ld << ")" << std::endl;
        return 1;
    }
    std::cout << " ok.\n";

    // 7) Transform check: ensure transform runs and errors are consistent
    std::cout << "7) Transform check (order=2)..." << std::flush;
    auto res_ref = act.transform(test_signal, 2);
    auto res_ld  = act_loaded->transform(test_signal, 2);
    double err_diff = std::fabs(res_ref.error - res_ld.error);
    if (err_diff > 1e-6) {
        std::cerr << "\nERROR: Transform error mismatch after load. diff=" << err_diff << std::endl;
        return 1;
    }
    std::cout << " ok (error diff=" << err_diff << ").\n";

    // 8) Cleanup temp file
    std::remove(dict_path);

    std::cout << "\n✅ Dictionary save/load test passed successfully!\n";
    return 0;
}
