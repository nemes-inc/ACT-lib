#include "ACT_CPU.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

// BLAS headers
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

// ALGLIB for BFGS optimization
#include "alglib/alglib-cpp/src/optimization.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

ACT_CPU::ACT_CPU(double FS, int length, const ParameterRanges& ranges, bool verbose)
    : FS(FS), length(length), param_ranges(ranges), verbose(verbose), dict_mat(), param_mat(), dict_size(0) {
    if (verbose) {
        std::cout << "\n===============================================\n";
        std::cout << "INITIALIZING ACT_CPU (Eigen + CBLAS)\n";
        std::cout << "===============================================\n\n";
    }
}

ACT_CPU::~ACT_CPU() {
}

VectorXd ACT_CPU::time_vector_seconds() const {
    VectorXd t(length);
    double invFS = 1.0 / FS;
    for (int i = 0; i < length; ++i) t[i] = static_cast<double>(i) * invFS;
    return t;
}

VectorXd ACT_CPU::linspace(double start, double end, double step) const {
    std::vector<double> vals;
    if (step <= 0) return Eigen::VectorXd();
    for (double v = start; v <= end; v += step) vals.push_back(v);
    VectorXd out(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) out[i] = vals[i];
    return out;
}

VectorXd ACT_CPU::g(double tc, double fc, double logDt, double c) const {
    // Bounds and stability checks
    if (std::isnan(tc) || std::isnan(fc) || std::isnan(logDt) || std::isnan(c)) {
        return VectorXd::Zero(length);
    }
    logDt = std::max(-10.0, std::min(2.0, logDt));

    // Convert tc from samples to seconds
    double tc_sec = tc / FS;

    double Dt = std::exp(logDt);
    if (Dt < 1e-10 || Dt > 100.0) return VectorXd::Zero(length);

    VectorXd t = time_vector_seconds();
    VectorXd time_diff = t.array() - tc_sec;

    // Gaussian window
    VectorXd exponent = (-0.5 * (time_diff.array() / Dt).square()).matrix();
    for (int i = 0; i < exponent.size(); ++i) if (exponent[i] < -50.0) exponent[i] = -50.0;

    VectorXd gaussian_window(exponent.size());
#ifdef __APPLE__
    {
        int n = exponent.size();
        vvexp(gaussian_window.data(), exponent.data(), &n);
    }
#else
    gaussian_window = exponent.array().exp().matrix();
#endif

    // Phase and cosine
    const double two_pi = 2.0 * M_PI;
    VectorXd phase = (two_pi * (c * time_diff.array().square() + fc * time_diff.array())).matrix();

    VectorXd complex_exp(phase.size());
#ifdef __APPLE__
    {
        int n = phase.size();
        vvcos(complex_exp.data(), phase.data(), &n);
    }
#else
    complex_exp = phase.array().cos().matrix();
#endif

    VectorXd chirplet(length);
#ifdef __APPLE__
    vDSP_vmulD(gaussian_window.data(), 1, complex_exp.data(), 1, chirplet.data(), 1, length);
#else
    chirplet = gaussian_window.array() * complex_exp.array();
#endif

    // Replace invalid entries
    for (int i = 0; i < chirplet.size(); ++i) if (!std::isfinite(chirplet[i])) chirplet[i] = 0.0;

    // L2 normalize
    double energy = 0.0;
#ifdef __APPLE__
    vDSP_dotprD(chirplet.data(), 1, chirplet.data(), 1, &energy, length);
#else
    energy = cblas_ddot(length, chirplet.data(), 1, chirplet.data(), 1);
#endif
    if (energy > 0.0) {
        double inv_norm = 1.0 / std::sqrt(energy);
#ifdef __APPLE__
        vDSP_vsmulD(chirplet.data(), 1, &inv_norm, chirplet.data(), 1, length);
#else
        for (int i = 0; i < chirplet.size(); ++i) chirplet[i] *= inv_norm;
#endif
    } else {
        chirplet.setZero();
    }

    return chirplet;
}

int ACT_CPU::generate_chirplet_dictionary() {
    VectorXd tc_vals = linspace(param_ranges.tc_min, param_ranges.tc_max, param_ranges.tc_step);
    VectorXd fc_vals = linspace(param_ranges.fc_min, param_ranges.fc_max, param_ranges.fc_step);
    VectorXd logDt_vals = linspace(param_ranges.logDt_min, param_ranges.logDt_max, param_ranges.logDt_step);
    VectorXd c_vals = linspace(param_ranges.c_min, param_ranges.c_max, param_ranges.c_step);

    dict_size = static_cast<int>(tc_vals.size() * fc_vals.size() * logDt_vals.size() * c_vals.size());

    if (verbose) {
        std::cout << "Dictionary length: " << dict_size << std::endl;
        std::cout << "Parameter ranges:\n";
        std::cout << "  tc: " << tc_vals.size() << " values (" << param_ranges.tc_min << " to " << param_ranges.tc_max << ")\n";
        std::cout << "  fc: " << fc_vals.size() << " values (" << param_ranges.fc_min << " to " << param_ranges.fc_max << ")\n";
        std::cout << "  logDt: " << logDt_vals.size() << " values (" << param_ranges.logDt_min << " to " << param_ranges.logDt_max << ")\n";
        std::cout << "  c: " << c_vals.size() << " values (" << param_ranges.c_min << " to " << param_ranges.c_max << ")\n";
    }

    dict_mat.resize(length, dict_size);
    param_mat.resize(dict_size, 4);

    int cnt = 0;
    int slow_cnt = 1;
    for (int itc = 0; itc < tc_vals.size(); ++itc) {
        if (verbose) {
            std::cout << "\n" << slow_cnt << "/" << tc_vals.size() << ": \t";
            slow_cnt++;
        }
        for (int ifc = 0; ifc < fc_vals.size(); ++ifc) {
            if (verbose) std::cout << ".";
            for (int ilog = 0; ilog < logDt_vals.size(); ++ilog) {
                for (int ic = 0; ic < c_vals.size(); ++ic) {
                    double tc = tc_vals[itc];
                    double fc = fc_vals[ifc];
                    double logDt = logDt_vals[ilog];
                    double c = c_vals[ic];
                    VectorXd atom = g(tc, fc, logDt, c);
                    dict_mat.col(cnt) = atom;
                    param_mat(cnt, 0) = tc;
                    param_mat(cnt, 1) = fc;
                    param_mat(cnt, 2) = logDt;
                    param_mat(cnt, 3) = c;
                    ++cnt;
                }
            }
        }
    }

    return dict_size;
}

std::pair<int,double> ACT_CPU::search_dictionary(const Eigen::Ref<const VectorXd>& signal) const {
    assert(signal.size() == length);
    if (dict_size == 0) return {0, 0.0};

    VectorXd scores(dict_size);
    scores.setZero();

    const int m = length;
    const int n = dict_size;
    const double alpha = 1.0;
    const double beta = 0.0;

    // scores = A^T * x  (A is m x n, column-major)
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha,
                dict_mat.data(), m,
                signal.data(), 1,
                beta, scores.data(), 1);

    // Find argmax
    int best_idx = 0;
    double best_val = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        double v = scores[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    return {best_idx, best_val};
}

std::pair<int,double> ACT_CPU::search_dictionary(const std::vector<double>& sig) const {
    assert(static_cast<int>(sig.size()) == length);
    Eigen::Map<const VectorXd> x(sig.data(), length);
    return search_dictionary(x);
}

// Optimizer wrapper for ALGLIB (mirrors ACTOptimizer)
struct ACTCPU_Optimizer {
    const ACT_CPU* act;
    VectorXd signal;
    ACTCPU_Optimizer(const ACT_CPU* a, const VectorXd& s) : act(a), signal(s) {}
    static void func(const alglib::real_1d_array &x, double &f, void *ptr) {
        std::vector<double> params(4);
        for (int i = 0; i < 4; ++i) params[i] = x[i];
        auto* opt = reinterpret_cast<ACTCPU_Optimizer*>(ptr);
        f = opt->act->minimize_this(params, opt->signal);
    }
};

Eigen::VectorXd ACT_CPU::bfgs_optimize(const Eigen::Vector4d& initial_params,
                                       const Eigen::Ref<const VectorXd>& signal) const {
    try {
        alglib::minbcstate state;
        alglib::minbcreport rep;
        alglib::real_1d_array x, bndl, bndu;
        x.setlength(4); bndl.setlength(4); bndu.setlength(4);
        for (int i = 0; i < 4; ++i) x[i] = initial_params[i];

        double tc_step = param_ranges.tc_step;
        double fc_step = param_ranges.fc_step;
        double logDt_step = param_ranges.logDt_step;
        double c_step = param_ranges.c_step;

        bndl[0] = std::max(param_ranges.tc_min,     initial_params[0] - tc_step);
        bndu[0] = std::min(param_ranges.tc_max,     initial_params[0] + tc_step);
        bndl[1] = std::max(param_ranges.fc_min,     initial_params[1] - fc_step);
        bndu[1] = std::min(param_ranges.fc_max,     initial_params[1] + fc_step);
        bndl[2] = std::max(param_ranges.logDt_min,  initial_params[2] - logDt_step);
        bndu[2] = std::min(param_ranges.logDt_max,  initial_params[2] + logDt_step);
        bndl[3] = std::max(param_ranges.c_min,      initial_params[3] - c_step);
        bndu[3] = std::min(param_ranges.c_max,      initial_params[3] + c_step);

        // Clamp initial guess
        for (int i = 0; i < 4; ++i) {
            if (x[i] < bndl[i]) x[i] = bndl[i] + 1e-6;
            if (x[i] > bndu[i]) x[i] = bndu[i] - 1e-6;
        }

        ACTCPU_Optimizer opt(this, signal);
        alglib::minbccreatef(x, 1e-6, state);
        alglib::minbcsetbc(state, bndl, bndu);
        alglib::minbcsetcond(state, 1e-5, 0, 0, 100);
        alglib::minbcsetxrep(state, false);

        alglib::minbcoptimize(state, ACTCPU_Optimizer::func, nullptr, &opt);
        alglib::minbcresults(state, x, rep);

        if (rep.terminationtype > 0) {
            Eigen::Vector4d result;
            for (int i = 0; i < 4; ++i) result[i] = x[i];
            return result;
        }
    } catch (...) {
        // fall through to return initial
    }
    return initial_params;
}

ACT_CPU::TransformResult ACT_CPU::transform(const Eigen::Ref<const VectorXd>& signal,
                                            int order, double residual_threshold) const {
    if (dict_size == 0 || dict_mat.cols() != dict_size || param_mat.rows() != dict_size) {
        throw std::runtime_error("ACT_CPU::transform called without a ready dictionary.");
    }

    TransformResult result;
    result.params = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>(order, 4);
    result.coeffs = VectorXd::Zero(order);
    result.signal = signal;
    result.approx = VectorXd::Zero(length);
    result.residue = signal;

    double prev_resid_norm2 = std::numeric_limits<double>::max();

    for (int i = 0; i < order; ++i) {
        // Coarse search via GEMV
        auto [ind, val] = search_dictionary(result.residue);

        // Coarse params
        Eigen::Vector4d init;
        init[0] = param_mat(ind, 0);
        init[1] = param_mat(ind, 1);
        init[2] = param_mat(ind, 2);
        init[3] = param_mat(ind, 3);

        // Refine with BFGS
        Eigen::Vector4d refined = bfgs_optimize(init, result.residue);

        // Atom and coefficient
        VectorXd atom = g(refined[0], refined[1], refined[2], refined[3]);

        double coeff = 0.0;
#ifdef __APPLE__
        vDSP_dotprD(atom.data(), 1, result.residue.data(), 1, &coeff, length);
#else
        coeff = cblas_ddot(length, atom.data(), 1, result.residue.data(), 1);
#endif

        // Store
        result.params.row(i) << refined[0], refined[1], refined[2], refined[3];
        result.coeffs[i] = coeff;

        // Update residue: residue -= coeff * atom
        double alpha_neg = -coeff;
#ifdef __APPLE__
        // Use cblas_daxpy from Accelerate
        cblas_daxpy(length, alpha_neg, atom.data(), 1, result.residue.data(), 1);
#else
        cblas_daxpy(length, alpha_neg, atom.data(), 1, result.residue.data(), 1);
#endif

        // Update approx: approx += coeff * atom
        double alpha_pos = coeff;
#ifdef __APPLE__
        cblas_daxpy(length, alpha_pos, atom.data(), 1, result.approx.data(), 1);
#else
        cblas_daxpy(length, alpha_pos, atom.data(), 1, result.approx.data(), 1);
#endif

        // Residual norm2
        double resid_norm2 = 0.0;
#ifdef __APPLE__
        vDSP_dotprD(result.residue.data(), 1, result.residue.data(), 1, &resid_norm2, length);
#else
        resid_norm2 = cblas_ddot(length, result.residue.data(), 1, result.residue.data(), 1);
#endif

        if (prev_resid_norm2 - resid_norm2 < residual_threshold) {
            // shrink matrices/vectors to i+1
            result.params.conservativeResize(i+1, 4);
            result.coeffs.conservativeResize(i+1);
            break;
        }
        prev_resid_norm2 = resid_norm2;
    }

    // Final error as L2 norm of residue
    double resid_norm2 = 0.0;
#ifdef __APPLE__
    vDSP_dotprD(result.residue.data(), 1, result.residue.data(), 1, &resid_norm2, length);
#else
    resid_norm2 = cblas_ddot(length, result.residue.data(), 1, result.residue.data(), 1);
#endif
    result.error = std::sqrt(resid_norm2);

    return result;
}

ACT_CPU::TransformResult ACT_CPU::transform(const std::vector<double>& sig,
                                            int order, double residual_threshold) const {
    assert(static_cast<int>(sig.size()) == length);
    Eigen::Map<const VectorXd> x(sig.data(), length);
    return transform(x, order, residual_threshold);
}

double ACT_CPU::minimize_this(const std::vector<double>& params,
                              const Eigen::Ref<const VectorXd>& signal) const {
    for (double p : params) {
        if (std::isnan(p) || std::isinf(p)) return 1e10;
    }
    VectorXd atom = g(params[0], params[1], params[2], params[3]);
    if (atom.size() == 0) return 1e10;
    double prod = 0.0;
#ifdef __APPLE__
    vDSP_dotprD(atom.data(), 1, signal.data(), 1, &prod, length);
#else
    prod = cblas_ddot(length, atom.data(), 1, signal.data(), 1);
#endif
    if (!std::isfinite(prod)) return 1e10;
    double result = -std::abs(prod);
    if (!std::isfinite(result)) return 1e10;
    return result;
}

bool ACT_CPU::save_dictionary(const std::string& file_path) const {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) return false;

    // Magic and version
    const char magic[8] = {'A','C','T','D','I','C','T','\0'};
    uint32_t version = 2;
    file.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Core parameters
    file.write(reinterpret_cast<const char*>(&FS), sizeof(FS));
    int32_t len32 = static_cast<int32_t>(length);
    file.write(reinterpret_cast<const char*>(&len32), sizeof(len32));
    uint8_t complex_mode_u8 = 0; // always real in ACT_CPU
    file.write(reinterpret_cast<const char*>(&complex_mode_u8), sizeof(complex_mode_u8));

    // ParameterRanges
    const double pr_vals[12] = {param_ranges.tc_min, param_ranges.tc_max, param_ranges.tc_step,
                                param_ranges.fc_min, param_ranges.fc_max, param_ranges.fc_step,
                                param_ranges.logDt_min, param_ranges.logDt_max, param_ranges.logDt_step,
                                param_ranges.c_min, param_ranges.c_max, param_ranges.c_step};
    file.write(reinterpret_cast<const char*>(pr_vals), sizeof(pr_vals));

    // Dictionary size
    int32_t ds32 = static_cast<int32_t>(dict_size);
    file.write(reinterpret_cast<const char*>(&ds32), sizeof(ds32));

    // dict_mat: write columns as contiguous length blocks to match ACT format
    VectorXd tmp(length);
    for (int i = 0; i < dict_size; ++i) {
        tmp = dict_mat.col(i);
        file.write(reinterpret_cast<const char*>(tmp.data()), length * sizeof(double));
    }

    // param_mat: rows of 4 doubles
    for (int i = 0; i < dict_size; ++i) {
        double row[4] = {param_mat(i,0), param_mat(i,1), param_mat(i,2), param_mat(i,3)};
        file.write(reinterpret_cast<const char*>(row), 4 * sizeof(double));
    }

    return file.good();
}

bool ACT_CPU::load_dictionary_data_from_stream(std::istream& file, ACT_CPU& instance) {
    char magic[8] = {0};
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (std::strncmp(magic, "ACTDICT", 7) != 0 || version < 1) return false;

    double FS_read = 0.0; int32_t len32 = 0; uint8_t complex_u8 = 0;
    file.read(reinterpret_cast<char*>(&FS_read), sizeof(FS_read));
    file.read(reinterpret_cast<char*>(&len32), sizeof(len32));
    file.read(reinterpret_cast<char*>(&complex_u8), sizeof(complex_u8));

    double pr_vals[12] = {0};
    file.read(reinterpret_cast<char*>(pr_vals), sizeof(pr_vals));
    ParameterRanges pr(
        pr_vals[0], pr_vals[1], pr_vals[2],
        pr_vals[3], pr_vals[4], pr_vals[5],
        pr_vals[6], pr_vals[7], pr_vals[8],
        pr_vals[9], pr_vals[10], pr_vals[11]
    );

    int32_t ds32 = 0;
    file.read(reinterpret_cast<char*>(&ds32), sizeof(ds32));
    if (ds32 < 0) return false;

    instance.FS = FS_read;
    instance.length = static_cast<int>(len32);
    instance.param_ranges = pr;
    instance.dict_size = ds32;
    instance.dict_mat.resize(instance.length, instance.dict_size);
    instance.param_mat.resize(instance.dict_size, 4);

    // Read dict columns
    VectorXd tmp(instance.length);
    for (int i = 0; i < instance.dict_size; ++i) {
        file.read(reinterpret_cast<char*>(tmp.data()), instance.length * sizeof(double));
        instance.dict_mat.col(i) = tmp;
    }
    // Read param rows
    for (int i = 0; i < instance.dict_size; ++i) {
        double row[4];
        file.read(reinterpret_cast<char*>(row), 4 * sizeof(double));
        instance.param_mat(i,0) = row[0];
        instance.param_mat(i,1) = row[1];
        instance.param_mat(i,2) = row[2];
        instance.param_mat(i,3) = row[3];
    }

    return file.good();
}
