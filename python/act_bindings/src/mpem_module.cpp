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
 
 #include "ACT.h"

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

class ActEngine {
public:
    ActEngine(double fs, int length, py::object ranges = py::none(),
              bool complex_mode = false, bool force_regenerate = false,
              bool mute = true, std::string dict_cache_file = "dict_cache.bin")
        : fs_(fs), length_(length) {
        ACT::ParameterRanges r = parse_ranges(ranges, fs, length);
        act_ = std::make_unique<ACT>(fs, length, dict_cache_file, r,
                                     complex_mode, force_regenerate, mute);
    }

    py::dict transform(py::array_t<double, py::array::c_style | py::array::forcecast> signal,
                       int order = 1, bool debug = false) {
        if (order <= 0) {
            throw std::invalid_argument("order must be a positive integer");
        }
        py::buffer_info info = signal.request();
        if (info.ndim != 1) {
            throw std::invalid_argument("signal must be a 1D NumPy array");
        }
        const int n = static_cast<int>(info.shape[0]);
        if (n != length_) {
            throw std::invalid_argument("signal length must equal engine length");
        }
        const double* ptr = static_cast<const double*>(info.ptr);
        std::vector<double> sig_vec(ptr, ptr + n);
        ACT::TransformResult tr = act_->transform(sig_vec, order, debug);

        py::dict out;
        if (tr.params.empty() || tr.coeffs.empty()) {
            out["frequency"] = py::float_(NAN);
            out["chirp_rate"] = py::float_(NAN);
            out["amplitude"] = py::float_(NAN);
            out["duration"] = py::float_(NAN);
            out["time_center"] = py::float_(NAN);
            out["spectral_width"] = py::float_(NAN);
            return out;
        }
        const std::vector<double>& p = tr.params[0];
        if (p.size() < 4) {
            out["frequency"] = py::float_(NAN);
            out["chirp_rate"] = py::float_(NAN);
            out["amplitude"] = py::float_(NAN);
            out["duration"] = py::float_(NAN);
            out["time_center"] = py::float_(NAN);
            out["spectral_width"] = py::float_(NAN);
            return out;
        }

        const double tc_samples = p[0];
        const double fc = p[1];
        const double logDt = p[2];
        const double c = p[3];
        const double coeff = tr.coeffs[0];

        const double Dt = std::exp(logDt);
        const double spec_w = 1.0 / (2.0 * kPI * std::max(Dt, 1e-12));

        out["frequency"] = py::float_(fc);
        out["chirp_rate"] = py::float_(c);
        out["amplitude"] = py::float_(std::fabs(coeff));
        out["duration"] = py::float_(Dt);
        out["time_center"] = py::float_(tc_samples / fs_);
        out["spectral_width"] = py::float_(spec_w);

        return out;
    }

    double fs() const { return fs_; }
    int length() const { return length_; }

private:
    double fs_;
    int length_;
    std::unique_ptr<ACT> act_;
};


} // namespace

PYBIND11_MODULE(mpem, m) {
    m.doc() = "Pybind11 bindings for ACT MPEM extraction";
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
             "Run ACT transform and return top-1 chirplet parameters.",
             py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def_property_readonly("fs", &ActEngine::fs)
        .def_property_readonly("length", &ActEngine::length);
}
