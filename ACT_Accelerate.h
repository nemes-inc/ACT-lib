#ifndef ACT_ACCELERATE_H
#define ACT_ACCELERATE_H

#include "ACT_CPU.h"
#include <iostream>

// Apple Accelerate-optimized subclass of ACT_CPU_T
// Uses vDSP/vForce (vvexp/vvcos) for faster chirplet generation and vector ops
// Falls back to base implementations on non-Apple platforms
template <typename Scalar>
class ACT_Accelerate_T : public ACT_CPU_T<Scalar> {
public:
    using Base = ACT_CPU_T<Scalar>;
    using ParameterRanges = typename Base::ParameterRanges;

    ACT_Accelerate_T(double FS, int length,
                     const ParameterRanges& ranges,
                     bool verbose = false)
        : Base(FS, length, ranges, verbose) {
        if (verbose) {
#ifdef __APPLE__
            std::cout << "[ACT_Accelerate] vDSP/vForce enabled; BLAS via Accelerate" << std::endl;
#else
            std::cout << "[ACT_Accelerate] Non-Apple platform detected: falling back to ACT_CPU implementations" << std::endl;
#endif
        }
    }

    // Override chirplet generation to use Accelerate when available
    act::VecX<Scalar> g(double tc, double fc, double logDt, double c) const override;

protected:
    Scalar dot(const Scalar* a, const Scalar* b, int n) const override;
    void axpy(int n, Scalar alpha, const Scalar* x, int incx, Scalar* y, int incy) const override;
};

// Default double-precision alias for compatibility
using ACT_Accelerate = ACT_Accelerate_T<double>;
using ACT_Accelerate_f = ACT_Accelerate_T<float>;

#endif // ACT_ACCELERATE_H
