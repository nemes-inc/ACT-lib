#include "ACT_Accelerate.h"

#include <cmath>
#include <algorithm>
#include <type_traits>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// g(): templated Accelerate implementation
template <typename Scalar>
act::VecX<Scalar> ACT_Accelerate_T<Scalar>::g(double tc, double fc, double logDt, double c) const {
#ifndef __APPLE__
    // Fallback to base implementation on non-Apple platforms
    return ACT_CPU_T<Scalar>::g(tc, fc, logDt, c);
#else
    if (std::isnan(tc) || std::isnan(fc) || std::isnan(logDt) || std::isnan(c)) {
        return act::VecX<Scalar>::Zero(this->get_length());
    }
    logDt = std::max(-10.0, std::min(2.0, logDt));

    const int length = this->get_length();
    const double FS = this->get_FS();

    double tc_sec = tc / FS;
    double Dt = std::exp(logDt);
    if (Dt < 1e-10 || Dt > 100.0) return act::VecX<Scalar>::Zero(length);

    act::VecX<Scalar> t = this->time_vector_seconds();
    act::VecX<Scalar> time_diff = t.array() - Scalar(tc_sec);

    act::VecX<Scalar> exponent = (Scalar(-0.5) * (time_diff.array() / Scalar(Dt)).square()).matrix();
    for (int i = 0; i < exponent.size(); ++i) if (exponent[i] < Scalar(-50.0)) exponent[i] = Scalar(-50.0);

    act::VecX<Scalar> gaussian_window(exponent.size());
    {
        int n = static_cast<int>(exponent.size());
        if constexpr (std::is_same_v<Scalar, double>) {
            vvexp(gaussian_window.data(), exponent.data(), &n);
        } else {
            vvexpf(gaussian_window.data(), exponent.data(), &n);
        }
    }

    const Scalar two_pi = Scalar(2.0 * M_PI);
    act::VecX<Scalar> phase = (two_pi * (Scalar(c) * time_diff.array().square() + Scalar(fc) * time_diff.array())).matrix();

    act::VecX<Scalar> complex_exp(phase.size());
    {
        int n = static_cast<int>(phase.size());
        if constexpr (std::is_same_v<Scalar, double>) {
            vvcos(complex_exp.data(), phase.data(), &n);
        } else {
            vvcosf(complex_exp.data(), phase.data(), &n);
        }
    }

    act::VecX<Scalar> chirplet(length);
    if constexpr (std::is_same_v<Scalar, double>) {
        vDSP_vmulD(gaussian_window.data(), 1, complex_exp.data(), 1, chirplet.data(), 1, length);
    } else {
        vDSP_vmul(gaussian_window.data(), 1, complex_exp.data(), 1, chirplet.data(), 1, length);
    }

    for (int i = 0; i < chirplet.size(); ++i) if (!std::isfinite(chirplet[i])) chirplet[i] = Scalar(0);

    // L2 norm and normalize
    if constexpr (std::is_same_v<Scalar, double>) {
        double energy = 0.0;
        vDSP_dotprD(chirplet.data(), 1, chirplet.data(), 1, &energy, length);
        if (energy > 0.0) {
            double inv_norm = 1.0 / std::sqrt(energy);
            vDSP_vsmulD(chirplet.data(), 1, &inv_norm, chirplet.data(), 1, length);
        } else {
            chirplet.setZero();
        }
    } else {
        float energy = 0.0f;
        vDSP_dotpr(chirplet.data(), 1, chirplet.data(), 1, &energy, length);
        if (energy > 0.0f) {
            float inv_norm = 1.0f / std::sqrt(energy);
            vDSP_vsmul(chirplet.data(), 1, &inv_norm, chirplet.data(), 1, length);
        } else {
            chirplet.setZero();
        }
    }
    return chirplet;
#endif
}

template <typename Scalar>
Scalar ACT_Accelerate_T<Scalar>::dot(const Scalar* a, const Scalar* b, int n) const {
#ifndef __APPLE__
    return ACT_CPU_T<Scalar>::dot(a, b, n);
#else
    if constexpr (std::is_same_v<Scalar, double>) {
        double res = 0.0;
        vDSP_dotprD(a, 1, b, 1, &res, n);
        return res;
    } else {
        float res = 0.0f;
        vDSP_dotpr(a, 1, b, 1, &res, n);
        return res;
    }
#endif
}

template <typename Scalar>
void ACT_Accelerate_T<Scalar>::axpy(int n, Scalar alpha, const Scalar* x, int incx, Scalar* y, int incy) const {
#ifndef __APPLE__
    ACT_CPU_T<Scalar>::axpy(n, alpha, x, incx, y, incy);
#else
    if constexpr (std::is_same_v<Scalar, double>) {
        cblas_daxpy(n, alpha, x, incx, y, incy);
    } else {
        cblas_saxpy(n, alpha, x, incx, y, incy);
    }
#endif
}

// Explicit instantiation for double (default alias) and float
template class ACT_Accelerate_T<double>;
template class ACT_Accelerate_T<float>;
