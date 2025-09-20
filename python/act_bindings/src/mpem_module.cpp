#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

 #include <algorithm>
 #include <memory>
 
 #include "ACT_CPU.h"
 #include "ACT_MLX.h"

namespace py = pybind11;

namespace {

constexpr double kPI = 3.1415926535897932384626433832795;


ACT::ParameterRanges make_ranges(double fs, int length) {
    // Keep dictionary size modest to avoid large memory while still functional
    ACT::ParameterRanges r;
    r.tc_min = 0.0;
    r.tc_max = static_cast<double>(length - 1);
    // Aim for ~24 tc positions
    int n_tc = std::min(24, std::max(8, length / 12));
    int tc_step = std::max(1, (length - 1) / std::max(1, n_tc - 1));
    r.tc_step = static_cast<double>(tc_step);

    // Frequency range: 0.7 .. min(60, fs/2.5)
    r.fc_min = 0.7;
    r.fc_max = std::max(2.0, std::min(60.0, fs / 2.5));
    // ~24 values across the band
    int n_fc = 24;
    double fc_span = std::max(1e-6, r.fc_max - r.fc_min);
    r.fc_step = fc_span / static_cast<double>(std::max(1, n_fc - 1));

    // Duration (log scale): exp(logDt) ~ 0.05 .. 0.37 s
    r.logDt_min = -3.0;
    r.logDt_max = -1.0;
    r.logDt_step = 0.5; // ~5 values

    // Chirp rate: -10 .. 10 Hz/s, coarse grid
    r.c_min = -10.0;
    r.c_max = 10.0;
    r.c_step = 5.0; // [-10,-5,0,5,10]

    return r;
}

// Convert classic ACT ranges to CPU/MLX ranges
static ACT_CPU::ParameterRanges to_cpu_ranges(const ACT::ParameterRanges& s) {
    ACT_CPU::ParameterRanges d;
    d.tc_min = s.tc_min; d.tc_max = s.tc_max; d.tc_step = s.tc_step;
    d.fc_min = s.fc_min; d.fc_max = s.fc_max; d.fc_step = s.fc_step;
    d.logDt_min = s.logDt_min; d.logDt_max = s.logDt_max; d.logDt_step = s.logDt_step;
    d.c_min = s.c_min; d.c_max = s.c_max; d.c_step = s.c_step;
    return d;
}
static ACT_CPU_f::ParameterRanges to_mlx_ranges(const ACT::ParameterRanges& s) {
    ACT_CPU_f::ParameterRanges d;
    d.tc_min = s.tc_min; d.tc_max = s.tc_max; d.tc_step = s.tc_step;
    d.fc_min = s.fc_min; d.fc_max = s.fc_max; d.fc_step = s.fc_step;
    d.logDt_min = s.logDt_min; d.logDt_max = s.logDt_max; d.logDt_step = s.logDt_step;
    d.c_min = s.c_min; d.c_max = s.c_max; d.c_step = s.c_step;
    return d;
}

template <typename Scalar>
static std::vector<double> to_std_vector(const act::VecX<Scalar>& v) {
    std::vector<double> out(v.size());
    for (int i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v[i]);
    return out;
}
template <typename Scalar>
static std::vector<std::vector<double>> params_to_list(const act::ParamsMat<Scalar>& M) {
    std::vector<std::vector<double>> out(M.rows());
    for (int i = 0; i < M.rows(); ++i) {
        out[i] = { static_cast<double>(M(i,0)), static_cast<double>(M(i,1)),
                   static_cast<double>(M(i,2)), static_cast<double>(M(i,3)) };
    }
    return out;
}

// Parse optional Python dict 'ranges' into ACT::ParameterRanges; falls back to defaults
ACT::ParameterRanges parse_ranges(py::object ranges_obj, double fs, int length) {
    ACT::ParameterRanges r = make_ranges(fs, length);
    if (ranges_obj.is_none()) {
        return r;
    }
    if (!py::isinstance<py::dict>(ranges_obj)) {
        throw std::invalid_argument("ranges must be a dict or None");
    }
    py::dict d = py::reinterpret_borrow<py::dict>(ranges_obj);
    auto set_if_present = [&](const char* key, double& field) {
        if (d.contains(key)) {
            field = py::cast<double>(d[key]);
        }
    };
    set_if_present("tc_min", r.tc_min);
    set_if_present("tc_max", r.tc_max);
    set_if_present("tc_step", r.tc_step);
    set_if_present("fc_min", r.fc_min);
    set_if_present("fc_max", r.fc_max);
    set_if_present("fc_step", r.fc_step);
    set_if_present("logDt_min", r.logDt_min);
    set_if_present("logDt_max", r.logDt_max);
    set_if_present("logDt_step", r.logDt_step);
    set_if_present("c_min", r.c_min);
    set_if_present("c_max", r.c_max);
    set_if_present("c_step", r.c_step);
    // Validate ranges
    if (r.tc_min < 0.0) throw std::invalid_argument("tc_min must be non-negative.");
    if (r.tc_max > length - 1) throw std::invalid_argument("tc_max must not exceed signal length - 1.");
    if (r.tc_min >= r.tc_max) throw std::invalid_argument("tc_min must be less than tc_max.");
    if (r.tc_step <= 0) throw std::invalid_argument("tc_step must be positive.");

    if (r.fc_min < 0.0) throw std::invalid_argument("fc_min must be non-negative.");
    if (r.fc_max > fs / 2.0) throw std::invalid_argument("fc_max must not exceed Nyquist frequency (fs/2).");
    if (r.fc_min >= r.fc_max) throw std::invalid_argument("fc_min must be less than fc_max.");
    if (r.fc_step <= 0) throw std::invalid_argument("fc_step must be positive.");

    if (r.logDt_min >= r.logDt_max) throw std::invalid_argument("logDt_min must be less than logDt_max.");
    if (r.logDt_step <= 0) throw std::invalid_argument("logDt_step must be positive.");

    if (r.c_min >= r.c_max) throw std::invalid_argument("c_min must be less than c_max.");
    if (r.c_step <= 0) throw std::invalid_argument("c_step must be positive.");
    return r;
}

class ActCPUEngine {
public:
    ActCPUEngine(double fs, int length, py::object ranges = py::none(),
                 bool force_regenerate = false, bool mute = true,
                 std::string dict_cache_file = std::string("dict_cpu.bin"))
        : fs_(fs), length_(length) {
        auto r_act = parse_ranges(ranges, fs, length);
        auto r = to_cpu_ranges(r_act);
        const bool verbose = !mute;
        if (!force_regenerate && !dict_cache_file.empty()) {
            auto loaded = ACT_CPU::load_dictionary<ACT_CPU>(dict_cache_file, verbose);
            if (loaded) { act_ = std::move(loaded); fs_ = act_->get_FS(); length_ = act_->get_length(); }
        }
        if (!act_) {
            act_ = std::make_unique<ACT_CPU>(fs, length, r, verbose);
            act_->generate_chirplet_dictionary();
            if (!dict_cache_file.empty()) act_->save_dictionary(dict_cache_file);
        }
    }

    py::dict transform(py::array_t<double, py::array::c_style | py::array::forcecast> signal,
                       int order = 1) const {
        if (order <= 0) throw std::invalid_argument("order must be a positive integer");
        py::buffer_info info = signal.request();
        if (info.ndim != 1) throw std::invalid_argument("signal must be a 1D NumPy array");
        const int n = static_cast<int>(info.shape[0]);
        if (n != length_) throw std::invalid_argument("signal length must equal engine length");
        const double* ptr = static_cast<const double*>(info.ptr);
        std::vector<double> sig_vec(ptr, ptr + n);
        auto tr = act_->transform(sig_vec, order);
        py::dict out;
        out["params"]  = py::cast(params_to_list(tr.params));
        out["coeffs"]  = py::cast(to_std_vector(tr.coeffs));
        out["signal"]  = py::cast(to_std_vector(tr.signal));
        out["approx"]  = py::cast(to_std_vector(tr.approx));
        out["residue"] = py::cast(to_std_vector(tr.residue));
        out["error"]   = py::float_(tr.error);
        return out;
    }

    double fs() const { return fs_; }
    int length() const { return length_; }
    py::dict dict_info() const {
        py::dict out; out["fs"] = fs_; out["length"] = length_;
        if (act_) {
            const auto& pr = act_->get_param_ranges();
            py::dict ranges;
            ranges["tc_min"] = pr.tc_min; ranges["tc_max"] = pr.tc_max; ranges["tc_step"] = pr.tc_step;
            ranges["fc_min"] = pr.fc_min; ranges["fc_max"] = pr.fc_max; ranges["fc_step"] = pr.fc_step;
            ranges["logDt_min"] = pr.logDt_min; ranges["logDt_max"] = pr.logDt_max; ranges["logDt_step"] = pr.logDt_step;
            ranges["c_min"] = pr.c_min; ranges["c_max"] = pr.c_max; ranges["c_step"] = pr.c_step;
            out["param_ranges"] = ranges; out["dict_size"] = act_->get_dict_size();
        }
        return out;
    }

private:
    double fs_;
    int length_;
    std::unique_ptr<ACT_CPU> act_;
};

class ActMLXEngine {
public:
    ActMLXEngine(double fs, int length, py::object ranges = py::none(),
                 bool force_regenerate = false, bool mute = true,
                 std::string dict_cache_file = std::string("dict_mlx.bin"))
        : fs_(fs), length_(length) {
        auto r_act = parse_ranges(ranges, fs, length);
        auto r = to_mlx_ranges(r_act);
        const bool verbose = !mute;
        if (!force_regenerate && !dict_cache_file.empty()) {
            auto loaded = ACT_CPU_f::load_dictionary<ACT_MLX_f>(dict_cache_file, verbose);
            if (loaded) { act_ = std::move(loaded); fs_ = act_->get_FS(); length_ = act_->get_length(); }
        }
        if (!act_) {
            act_ = std::make_unique<ACT_MLX_f>(fs, length, r, verbose);
            act_->generate_chirplet_dictionary();
            if (!dict_cache_file.empty()) act_->save_dictionary(dict_cache_file);
        }
    }

    py::dict transform(py::array signal, int order = 1) const {
        if (order <= 0) throw std::invalid_argument("order must be a positive integer");
        py::buffer_info info = signal.request();
        if (info.ndim != 1) throw std::invalid_argument("signal must be a 1D NumPy array");
        const int n = static_cast<int>(info.shape[0]);
        if (n != length_) throw std::invalid_argument("signal length must equal engine length");
        std::vector<float> sigf(n);
        if (info.format == py::format_descriptor<float>::format()) {
            const float* ptr = static_cast<const float*>(info.ptr);
            sigf.assign(ptr, ptr + n);
        } else if (info.format == py::format_descriptor<double>::format()) {
            const double* ptr = static_cast<const double*>(info.ptr);
            for (int i = 0; i < n; ++i) sigf[i] = static_cast<float>(ptr[i]);
        } else {
            throw std::invalid_argument("signal must be float32 or float64");
        }
        auto tr = act_->transform(sigf, order);
        py::dict out;
        out["params"]  = py::cast(params_to_list(tr.params));
        out["coeffs"]  = py::cast(to_std_vector(tr.coeffs));
        out["signal"]  = py::cast(to_std_vector(tr.signal));
        out["approx"]  = py::cast(to_std_vector(tr.approx));
        out["residue"] = py::cast(to_std_vector(tr.residue));
        out["error"]   = py::float_(tr.error);
        return out;
    }

    double fs() const { return fs_; }
    int length() const { return length_; }
    py::dict dict_info() const {
        py::dict out; out["fs"] = fs_; out["length"] = length_;
        if (act_) {
            const auto& pr = act_->get_param_ranges();
            py::dict ranges;
            ranges["tc_min"] = pr.tc_min; ranges["tc_max"] = pr.tc_max; ranges["tc_step"] = pr.tc_step;
            ranges["fc_min"] = pr.fc_min; ranges["fc_max"] = pr.fc_max; ranges["fc_step"] = pr.fc_step;
            ranges["logDt_min"] = pr.logDt_min; ranges["logDt_max"] = pr.logDt_max; ranges["logDt_step"] = pr.logDt_step;
            ranges["c_min"] = pr.c_min; ranges["c_max"] = pr.c_max; ranges["c_step"] = pr.c_step;
            out["param_ranges"] = ranges; out["dict_size"] = act_->get_dict_size();
        }
        return out;
    }

private:
    double fs_;
    int length_;
    std::unique_ptr<ACT_MLX_f> act_;
};


} // namespace

PYBIND11_MODULE(mpbfgs, m) {
    m.doc() = "PyACT bindings for ACT_CPU (double) and ACT_MLX (float32) engines.";

    py::class_<ActCPUEngine>(m, "ActCPUEngine")
        .def(py::init<double, int, py::object, bool, bool, std::string>(),
             py::arg("fs"),
             py::arg("length"),
             py::arg("ranges") = py::none(),
             py::arg("force_regenerate") = false,
             py::arg("mute") = true,
             py::arg("dict_cache_file") = std::string("dict_cpu.bin"))
        .def("transform", &ActCPUEngine::transform,
             py::arg("signal"),
             py::arg("order") = 1,
             py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def_property_readonly("fs", &ActCPUEngine::fs)
        .def_property_readonly("length", &ActCPUEngine::length)
        .def("dict_info", &ActCPUEngine::dict_info);

    py::class_<ActMLXEngine>(m, "ActMLXEngine")
        .def(py::init<double, int, py::object, bool, bool, std::string>(),
             py::arg("fs"),
             py::arg("length"),
             py::arg("ranges") = py::none(),
             py::arg("force_regenerate") = false,
             py::arg("mute") = true,
             py::arg("dict_cache_file") = std::string("dict_mlx.bin"))
        .def("transform", &ActMLXEngine::transform,
             py::arg("signal"),
             py::arg("order") = 1,
             py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def_property_readonly("fs", &ActMLXEngine::fs)
        .def_property_readonly("length", &ActMLXEngine::length)
        .def("dict_info", &ActMLXEngine::dict_info);

    // Backward-compatible wrapper: ActEngine -> defaults to CPU backend
    class ActEngine {
    public:
        ActEngine(double fs, int length, py::object ranges = py::none(),
                  bool /*complex_mode*/ = false,
                  bool force_regenerate = false,
                  bool mute = true,
                  std::string dict_cache_file = std::string("dict_cache.bin"))
            : cpu_(fs, length, ranges, force_regenerate, mute, dict_cache_file) {}

        py::dict transform(py::array_t<double, py::array::c_style | py::array::forcecast> signal,
                           int order = 1, bool /*debug*/ = false) const {
            return cpu_.transform(signal, order);
        }

        double fs() const { return cpu_.fs(); }
        int length() const { return cpu_.length(); }
        py::dict dict_info() const { return cpu_.dict_info(); }

    private:
        ActCPUEngine cpu_;
    };

    py::class_<ActEngine>(m, "ActEngine")
        .def(py::init<double, int, py::object, bool, bool, bool, std::string>(),
             py::arg("fs"),
             py::arg("length"),
             py::arg("ranges") = py::none(),
             py::arg("complex_mode") = false,
             py::arg("force_regenerate") = false,
             py::arg("mute") = true,
             py::arg("dict_cache_file") = std::string("dict_cache.bin"))
        .def("transform", &ActEngine::transform,
             py::arg("signal"),
             py::arg("order") = 1,
             py::arg("debug") = false,
             py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def_property_readonly("fs", &ActEngine::fs)
        .def_property_readonly("length", &ActEngine::length)
        .def("dict_info", &ActEngine::dict_info);
}
