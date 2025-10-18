#include "ACT_cuBLAS.h"

#include <iostream>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

// Destructor
template <typename Scalar>
ACT_cuBLAS_T<Scalar>::~ACT_cuBLAS_T() {
#ifdef USE_CUDA
    cleanup_cuda();
#endif
}

// search_dictionary: GPU fast path for float, else CPU fallback
template <typename Scalar>
std::pair<int, Scalar> ACT_cuBLAS_T<Scalar>::search_dictionary(
    const Eigen::Ref<const act::VecX<Scalar>>& signal) const {
    if (this->get_dict_size() == 0) return {0, Scalar(0)};

#ifdef USE_CUDA
    if constexpr (std::is_same_v<Scalar, float>) {
        ensure_cuda_resources();
        ensure_cuda_dict();

        const int m = this->get_length();
        const int n = this->get_dict_size();

        // Upload x (float32) directly on the same stream as cuBLAS
        cudaMemcpyAsync(d_x_, signal.data(), sizeof(float) * static_cast<size_t>(m), cudaMemcpyHostToDevice, stream_);

        const float alpha = 1.0f;
        const float beta = 0.0f;
        // scores = A^T * x  (A is m x n, column-major)
        cublasStatus_t st = cublasSgemv(handle_, CUBLAS_OP_T,
                                        m, n,
                                        &alpha,
                                        d_A_, m,
                                        d_x_, 1,
                                        &beta,
                                        d_scores_, 1);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasSgemv failed in ACT_cuBLAS::search_dictionary");
        }

        // Argmax by magnitude
        int best_idx_1based = 1;
        st = cublasIsamax(handle_, n, d_scores_, 1, &best_idx_1based);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasIsamax failed in ACT_cuBLAS::search_dictionary");
        }
        int best_idx = best_idx_1based - 1;

        // Fetch best value (signed)
        float best_val = 0.0f;
        cudaMemcpyAsync(&best_val, d_scores_ + best_idx, sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        return {best_idx, static_cast<Scalar>(best_val)};
    }
#endif
    // Fallback: CPU path
    return Base::search_dictionary(signal);
}

// generate_chirplet_dictionary: build on CPU, then upload and warm up
template <typename Scalar>
int ACT_cuBLAS_T<Scalar>::generate_chirplet_dictionary() {
    int n = Base::generate_chirplet_dictionary();
#ifdef USE_CUDA
    if constexpr (std::is_same_v<Scalar, float>) {
        ensure_cuda_resources();
        ensure_cuda_dict();
        warmup_kernels();
    }
#endif
    return n;
}

// on_dictionary_loaded: after loading from disk, upload and warm up
template <typename Scalar>
void ACT_cuBLAS_T<Scalar>::on_dictionary_loaded() {
#ifdef USE_CUDA
    if constexpr (std::is_same_v<Scalar, float>) {
        ensure_cuda_resources();
        ensure_cuda_dict();
        warmup_kernels();
    }
#endif
}

#ifdef USE_CUDA

// Ensure cuBLAS handle, stream, and reusable buffers exist
template <typename Scalar>
void ACT_cuBLAS_T<Scalar>::ensure_cuda_resources() const {
    if (!handle_) {
        cublasCreate(&handle_);
    }
    if (!stream_) {
        cudaStreamCreate(&stream_);
        cublasSetStream(handle_, stream_);
    }
}

// Pack dictionary to float32 column-major and upload to device; (re)allocate buffers
template <typename Scalar>
void ACT_cuBLAS_T<Scalar>::ensure_cuda_dict() const {
    if constexpr (std::is_same_v<Scalar, float>) {
        const int m = this->get_length();
        const int n = this->get_dict_size();

        bool need_pack = !cuda_ready_ || (m_ != m) || (n_ != n) || (d_A_ == nullptr);
        if (!need_pack) return;

        m_ = m; n_ = n;
        dict_colmajor_f32_.resize(static_cast<size_t>(m_) * static_cast<size_t>(n_));

        // Copy from Eigen column-major matrix to contiguous float buffer
        const auto& A = this->get_dict_mat();
        std::memcpy(dict_colmajor_f32_.data(), A.data(), sizeof(float) * static_cast<size_t>(m_) * static_cast<size_t>(n_));

        // Recreate device buffers
        if (d_A_) cudaFree(d_A_);
        if (d_x_) cudaFree(d_x_);
        if (d_scores_) cudaFree(d_scores_);

        cudaMalloc(reinterpret_cast<void**>(&d_A_), sizeof(float) * static_cast<size_t>(m_) * static_cast<size_t>(n_));
        cudaMalloc(reinterpret_cast<void**>(&d_x_), sizeof(float) * static_cast<size_t>(m_));
        cudaMalloc(reinterpret_cast<void**>(&d_scores_), sizeof(float) * static_cast<size_t>(n_));

        // Upload A on the same stream
        cudaMemcpyAsync(d_A_, dict_colmajor_f32_.data(), sizeof(float) * static_cast<size_t>(m_) * static_cast<size_t>(n_), cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);

        cuda_ready_ = true;
    } else {
        return;
    }
}

// Warmup: run a dummy GEMV and IAMAX to initialize kernels/runtime
template <typename Scalar>
void ACT_cuBLAS_T<Scalar>::warmup_kernels() const {
    if constexpr (std::is_same_v<Scalar, float>) {
        if (!cuda_ready_) return;

        // Zero x
        cudaMemsetAsync(d_x_, 0, sizeof(float) * static_cast<size_t>(m_), stream_);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemv(handle_, CUBLAS_OP_T, m_, n_, &alpha, d_A_, m_, d_x_, 1, &beta, d_scores_, 1);
        int idx = 1;
        cublasIsamax(handle_, n_, d_scores_, 1, &idx);
        cudaStreamSynchronize(stream_);
    } else {
        return;
    }
}

// Cleanup all CUDA resources
template <typename Scalar>
void ACT_cuBLAS_T<Scalar>::cleanup_cuda() const {
    if (d_A_) { cudaFree(d_A_); d_A_ = nullptr; }
    if (d_x_) { cudaFree(d_x_); d_x_ = nullptr; }
    if (d_scores_) { cudaFree(d_scores_); d_scores_ = nullptr; }
    if (handle_) { cublasDestroy(handle_); handle_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    cuda_ready_ = false;
}

#endif // USE_CUDA

// Explicit instantiation
template class ACT_cuBLAS_T<double>;
template class ACT_cuBLAS_T<float>;
