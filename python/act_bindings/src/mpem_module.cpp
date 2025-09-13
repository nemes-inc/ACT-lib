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
        const bool verbose = !mute;

        // Try to load dictionary from cache if not forcing regeneration
        if (!force_regenerate && !dict_cache_file.empty()) {
            auto loaded = ACT::load_dictionary<ACT>(dict_cache_file, verbose);
            if (loaded) {
                act_ = std::move(loaded);
                // Sync interface state with loaded dictionary
                fs_ = act_->get_FS();
                length_ = act_->get_length();
            }
        }

        // If not loaded, create a new ACT, generate dictionary, and optionally save it
        if (!act_) {
            act_ = std::make_unique<ACT>(fs, length, r, complex_mode, verbose);
            act_->generate_chirplet_dictionary();
            if (!dict_cache_file.empty()) {
                act_->save_dictionary(dict_cache_file);
            }
        }
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
        ACT::TransformResult tr = act_->transform(sig_vec, order);

        // Build full result dictionary
        py::dict out;
        out["params"]  = py::cast(tr.params);   // list[list[tc, fc, logDt, c]]
        out["coeffs"]  = py::cast(tr.coeffs);   // list[float]
        out["error"]   = py::float_(tr.error);  // float
        out["signal"]  = py::cast(tr.signal);   // list[float]
        out["approx"]  = py::cast(tr.approx);   // list[float]
        out["residue"] = py::cast(tr.residue);  // list[float]
        return out;
    }

    double fs() const { return fs_; }
    int length() const { return length_; }

    py::dict dict_info() const {
        py::dict out;
        out["fs"] = py::float_(fs_);
        out["length"] = py::int_(length_);
        if (act_) {
            const auto& pr = act_->get_param_ranges();
            py::dict ranges;
            ranges["tc_min"] = pr.tc_min;
            ranges["tc_max"] = pr.tc_max;
            ranges["tc_step"] = pr.tc_step;
            ranges["fc_min"] = pr.fc_min;
            ranges["fc_max"] = pr.fc_max;
            ranges["fc_step"] = pr.fc_step;
            ranges["logDt_min"] = pr.logDt_min;
            ranges["logDt_max"] = pr.logDt_max;
            ranges["logDt_step"] = pr.logDt_step;
            ranges["c_min"] = pr.c_min;
            ranges["c_max"] = pr.c_max;
            ranges["c_step"] = pr.c_step;
            out["param_ranges"] = ranges;
            out["dict_size"] = py::int_(act_->get_dict_size());
        }
        return out;
    }

private:
    double fs_;
    int length_;
    std::unique_ptr<ACT> act_;
};


} // namespace

PYBIND11_MODULE(mpbfgs, m) {
    m.doc() = "Pybind11 bindings for ACT MP-BFGS extraction";
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
        .def_property_readonly("length", &ActEngine::length)
        .def("dict_info", &ActEngine::dict_info,
             "Return dictionary metadata: fs, length, param_ranges, dict_size");
}
