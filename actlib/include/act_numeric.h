#ifndef ACT_NUMERIC_H
#define ACT_NUMERIC_H

// Thin, header-only BLAS wrappers for float/double.
// Centralizes precision-specific calls to enable future templating.

#include <type_traits>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace act {
namespace blas {

// dot
inline float dot(int n, const float* a, int inc_a, const float* b, int inc_b) {
    return cblas_sdot(n, a, inc_a, b, inc_b);
}
inline double dot(int n, const double* a, int inc_a, const double* b, int inc_b) {
    return cblas_ddot(n, a, inc_a, b, inc_b);
}

// axpy: y := alpha * x + y
inline void axpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    cblas_saxpy(n, alpha, x, incx, y, incy);
}
inline void axpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

// gemv (column-major): y := alpha * A^T * x + beta * y
inline void gemv_colmajor_trans(int m, int n,
                                float alpha,
                                const float* A, int lda,
                                const float* x, int incx,
                                float beta,
                                float* y, int incy) {
    cblas_sgemv(CblasColMajor, CblasTrans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
inline void gemv_colmajor_trans(int m, int n,
                                double alpha,
                                const double* A, int lda,
                                const double* x, int incx,
                                double beta,
                                double* y, int incy) {
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

// gemm (column-major): C := alpha * op(A) * op(B) + beta * C
inline void gemm_colmajor(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                          int M, int N, int K,
                          float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float beta,
                          float* C, int ldc) {
    cblas_sgemm(CblasColMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void gemm_colmajor(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                          int M, int N, int K,
                          double alpha,
                          const double* A, int lda,
                          const double* B, int ldb,
                          double beta,
                          double* C, int ldc) {
    cblas_dgemm(CblasColMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// iamax: index of the element with maximum absolute value
inline int iamax(int n, const float* x, int incx) {
    return static_cast<int>(cblas_isamax(n, x, incx));
}
inline int iamax(int n, const double* x, int incx) {
    return static_cast<int>(cblas_idamax(n, x, incx));
}

} // namespace blas
} // namespace act

#endif // ACT_NUMERIC_H
