#include "ACT_CUDA_TK.h"

#include <iostream>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <climits>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#if defined(USE_CUDA) && defined(USE_TK)
#include "kittens.cuh"
#endif

#ifdef USE_CUDA

#if defined(USE_TK)
// TK-tiled FP32 GEMV: each block processes TN_TILE columns; tiles K in chunks of TK_TILE
// Additionally computes per-block argmax(|y|) and writes it to blk_best_* arrays.
static __global__ void gemv_tk_kernel(const float* __restrict__ A, // (m x n) column-major
                                      const float* __restrict__ x, // (m)
                                      float* __restrict__ y,       // (n)
                                      int m, int n,
                                      float* __restrict__ blk_best_abs, // (grid.x)
                                      float* __restrict__ blk_best_val, // (grid.x)
                                      int*   __restrict__ blk_best_idx) // (grid.x)
{
    // Tile sizes (multiples of 16 for TK shared tiles)
    constexpr int TK_TILE = 64;   // rows (K)
    constexpr int TN_TILE = 128;  // cols (N)

    // One thread per column in the tile
    int j0 = blockIdx.x * TN_TILE;
    int t  = threadIdx.x; // local column in tile
    if (t >= TN_TILE) return;
    int col = j0 + t;

    // Shared tile for A block (TK_TILE x TN_TILE)
    __shared__ kittens::st_fl<TK_TILE, TN_TILE> As;
    // Shared tile for x slice (TK_TILE)
    __shared__ float Xs[TK_TILE];

    float acc = 0.0f;

    // Sweep K dimension in TK_TILE chunks
    for (int k0 = 0; k0 < m; k0 += TK_TILE) {
        // Cooperative load of A tile into shared (row-major indices)
        int total = TK_TILE * TN_TILE;
        for (int idx = t; idx < total; idx += blockDim.x) {
            int r = idx % TK_TILE;      // row within tile
            int c = idx / TK_TILE;      // col within tile
            int j = j0 + c;             // global column

            float val = 0.0f;
            int i = k0 + r;             // global row
            if (j < n && i < m) {
                val = A[static_cast<size_t>(j) * static_cast<size_t>(m) + i];
            }
            As[{r, c}] = val;
        }
        __syncthreads();

        // Load x tile once per block (first TK_TILE threads)
        if (t < TK_TILE) {
            int i_tile = k0 + t;
            Xs[t] = (i_tile < m) ? x[i_tile] : 0.0f;
        }
        __syncthreads();

        // Accumulate dot for this column using shared x
        if (col < n) {
            #pragma unroll
            for (int r = 0; r < TK_TILE; ++r) {
                int i = k0 + r;
                if (i < m) {
                    float a = As[{r, t}];
                    acc += a * Xs[r];
                }
            }
        }
        __syncthreads();
    }

    if (col < n) {
        y[col] = acc;
    }

    // ---------------- Per-block argmax(|y|) reduction ----------------
    // Each thread starts with its candidate (abs(acc), acc, col). Threads with col>=n are invalid.
    float best_abs = (col < n) ? fabsf(acc) : -INFINITY;
    float best_val = acc;
    int   best_idx = col;

    // Warp-level reduction using shuffles
    unsigned mask = 0xffffffffu;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5; // 32 threads per warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o_abs = __shfl_down_sync(mask, best_abs, offset);
        float o_val = __shfl_down_sync(mask, best_val, offset);
        int   o_idx = __shfl_down_sync(mask, best_idx, offset);
        if (o_abs > best_abs || (o_abs == best_abs && o_idx < best_idx)) {
            best_abs = o_abs;
            best_val = o_val;
            best_idx = o_idx;
        }
    }

    // One tuple per warp into shared memory
    __shared__ float warp_best_abs[32];
    __shared__ float warp_best_val[32];
    __shared__ int   warp_best_idx[32];
    if (lane == 0) {
        warp_best_abs[warp_id] = best_abs;
        warp_best_val[warp_id] = best_val;
        warp_best_idx[warp_id] = best_idx;
    }
    __syncthreads();

    // Cross-warp reduction by warp 0
    const int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        best_abs = (lane < num_warps) ? warp_best_abs[lane] : -INFINITY;
        best_val = (lane < num_warps) ? warp_best_val[lane] : 0.0f;
        best_idx = (lane < num_warps) ? warp_best_idx[lane] : INT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o_abs = __shfl_down_sync(mask, best_abs, offset);
            float o_val = __shfl_down_sync(mask, best_val, offset);
            int   o_idx = __shfl_down_sync(mask, best_idx, offset);
            if (o_abs > best_abs || (o_abs == best_abs && o_idx < best_idx)) {
                best_abs = o_abs;
                best_val = o_val;
                best_idx = o_idx;
            }
        }
        if (lane == 0) {
            blk_best_abs[blockIdx.x] = best_abs;
            blk_best_val[blockIdx.x] = best_val;
            blk_best_idx[blockIdx.x] = best_idx;
        }
    }
}

// Reduce per-block winners to a single global winner
static __global__ void reduce_block_bests_kernel(const float* __restrict__ blk_best_abs,
                                                 const float* __restrict__ blk_best_val,
                                                 const int*   __restrict__ blk_best_idx,
                                                 int count,
                                                 float* __restrict__ out_best_val,
                                                 int*   __restrict__ out_best_idx) {
    // Each thread reduces a strided subset, then we do a block-wide reduction
    float best_abs = -INFINITY;
    float best_val = 0.0f;
    int   best_idx = INT_MAX;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        float a = blk_best_abs[i];
        float v = blk_best_val[i];
        int   j = blk_best_idx[i];
        if (a > best_abs || (a == best_abs && j < best_idx)) {
            best_abs = a; best_val = v; best_idx = j;
        }
    }

    // Warp reduction
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o_abs = __shfl_down_sync(mask, best_abs, offset);
        float o_val = __shfl_down_sync(mask, best_val, offset);
        int   o_idx = __shfl_down_sync(mask, best_idx, offset);
        if (o_abs > best_abs || (o_abs == best_abs && o_idx < best_idx)) {
            best_abs = o_abs;
            best_val = o_val;
            best_idx = o_idx;
        }
    }

    // Cross-warp reduction via shared memory
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ float warp_best_abs[32];
    __shared__ float warp_best_val[32];
    __shared__ int   warp_best_idx[32];
    if (lane == 0) {
        warp_best_abs[warp_id] = best_abs;
        warp_best_val[warp_id] = best_val;
        warp_best_idx[warp_id] = best_idx;
    }
    __syncthreads();
    const int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        best_abs = (lane < num_warps) ? warp_best_abs[lane] : -INFINITY;
        best_val = (lane < num_warps) ? warp_best_val[lane] : 0.0f;
        best_idx = (lane < num_warps) ? warp_best_idx[lane] : INT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float o_abs = __shfl_down_sync(mask, best_abs, offset);
            float o_val = __shfl_down_sync(mask, best_val, offset);
            int   o_idx = __shfl_down_sync(mask, best_idx, offset);
            if (o_abs > best_abs || (o_abs == best_abs && o_idx < best_idx)) {
                best_abs = o_abs;
                best_val = o_val;
                best_idx = o_idx;
            }
        }
        if (lane == 0) {
            *out_best_val = best_val;
            *out_best_idx = best_idx;
        }
    }
}
#else
// Fallback CUDA kernel: one column per block with shared-memory reduction
static __global__ void gemv_cols_kernel(const float* __restrict__ A, // (m x n) column-major
                                        const float* __restrict__ x, // (m)
                                        float* __restrict__ y,       // (n)
                                        int m, int n, int maxBlocksX) {
    int j = blockIdx.x + blockIdx.y * maxBlocksX; // column index
    if (j >= n) return;

    const float* col = A + static_cast<size_t>(j) * static_cast<size_t>(m);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        sum += col[i] * x[i];
    }

    __shared__ float shm[256]; // assume blockDim.x <= 256
    shm[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[j] = shm[0];
    }
}
#endif // USE_TK

#endif // USE_CUDA

// Destructor
template <typename Scalar>
ACT_CUDA_TK_T<Scalar>::~ACT_CUDA_TK_T() {
#ifdef USE_CUDA
    cleanup_cuda();
#endif
}

// search_dictionary: GPU custom kernel for float, else CPU fallback
template <typename Scalar>
std::pair<int, Scalar> ACT_CUDA_TK_T<Scalar>::search_dictionary(
    const Eigen::Ref<const act::VecX<Scalar>>& signal) const {
    if (this->get_dict_size() == 0) return {0, Scalar(0)};

#ifdef USE_CUDA
    if constexpr (std::is_same_v<Scalar, float>) {
        ensure_cuda_resources();
        ensure_cuda_dict();

        const int m = this->get_length();
        const int n = this->get_dict_size();

        // Upload x (float32) on stream
        cudaMemcpyAsync(d_x_, signal.data(), sizeof(float) * static_cast<size_t>(m), cudaMemcpyHostToDevice, stream_);

        // Launch GEMV kernel: y = A^T x
        #if defined(USE_TK)
        {
            constexpr int TN_TILE = 128;
            int gridX = (n + TN_TILE - 1) / TN_TILE;
            dim3 grid(gridX, 1, 1);
            dim3 block(TN_TILE, 1, 1);
            // GEMV + per-block argmax
            gemv_tk_kernel<<<grid, block, 0, stream_>>>(
                d_A_, d_x_, d_scores_, m, n,
                d_blk_best_abs_, d_blk_best_val_, d_blk_best_idx_);
            // Final reduction of per-block winners
            reduce_block_bests_kernel<<<1, 128, 0, stream_>>>(
                d_blk_best_abs_, d_blk_best_val_, d_blk_best_idx_, gridX,
                d_final_best_val_, d_final_best_idx_);

            // Fetch final best only
            float best_val_h = 0.0f; int best_idx_h = 0;
            cudaMemcpyAsync(&best_val_h, d_final_best_val_, sizeof(float), cudaMemcpyDeviceToHost, stream_);
            cudaMemcpyAsync(&best_idx_h, d_final_best_idx_, sizeof(int),   cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
            return {best_idx_h, static_cast<Scalar>(best_val_h)};
        }
        #else
        {
            const int MAX_BLOCKS_X = 65535;
            int blocksX = (n < MAX_BLOCKS_X) ? n : MAX_BLOCKS_X;
            int blocksY = (n + MAX_BLOCKS_X - 1) / MAX_BLOCKS_X;
            dim3 grid(blocksX > 0 ? blocksX : 1, blocksY > 0 ? blocksY : 1, 1);
            dim3 block(256, 1, 1);
            gemv_cols_kernel<<<grid, block, 0, stream_>>>(d_A_, d_x_, d_scores_, m, n, MAX_BLOCKS_X);
        }
        #endif
        // Copy scores back and compute argmax by magnitude on host (fallback branch)
        std::vector<float> host_scores(n);
        cudaMemcpyAsync(host_scores.data(), d_scores_, sizeof(float) * static_cast<size_t>(n), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        int best_idx = 0;
        float best_val = host_scores[0];
        float best_abs = std::fabs(best_val);
        for (int j = 1; j < n; ++j) {
            float v = host_scores[j];
            float a = std::fabs(v);
            if (a > best_abs) { best_abs = a; best_val = v; best_idx = j; }
        }
        return {best_idx, static_cast<Scalar>(best_val)};
    }
#endif
    // Fallback: CPU path
    return this->Base::search_dictionary(signal);
}

// generate_chirplet_dictionary: build on CPU, then upload and warm up
template <typename Scalar>
int ACT_CUDA_TK_T<Scalar>::generate_chirplet_dictionary() {
    int n = this->Base::generate_chirplet_dictionary();
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
void ACT_CUDA_TK_T<Scalar>::on_dictionary_loaded() {
#ifdef USE_CUDA
    if constexpr (std::is_same_v<Scalar, float>) {
        ensure_cuda_resources();
        ensure_cuda_dict();
        warmup_kernels();
    }
#endif
}

#ifdef USE_CUDA

// Ensure CUDA stream exists
template <typename Scalar>
void ACT_CUDA_TK_T<Scalar>::ensure_cuda_resources() const {
    if (!stream_) {
        cudaStreamCreate(&stream_);
    }
}

// Pack dictionary to float32 column-major and upload to device; (re)allocate buffers
template <typename Scalar>
void ACT_CUDA_TK_T<Scalar>::ensure_cuda_dict() const {
    if constexpr (std::is_same_v<Scalar, float>) {
        const int m = this->get_length();
        const int n = this->get_dict_size();

        bool need_pack = !cuda_ready_ || (m_ != m) || (n_ != n) || (d_A_ == nullptr);
        if (!need_pack) return;

        m_ = m; n_ = n;
        dict_colmajor_f32_.resize(static_cast<size_t>(m_) * static_cast<size_t>(n_));

        const auto& A = this->get_dict_mat();
        std::memcpy(dict_colmajor_f32_.data(), A.data(), sizeof(float) * static_cast<size_t>(m_) * static_cast<size_t>(n_));

        // Recreate device buffers
        if (d_A_) cudaFree(d_A_);
        if (d_x_) cudaFree(d_x_);
        if (d_scores_) cudaFree(d_scores_);
        if (d_blk_best_abs_) cudaFree(d_blk_best_abs_), d_blk_best_abs_ = nullptr;
        if (d_blk_best_val_) cudaFree(d_blk_best_val_), d_blk_best_val_ = nullptr;
        if (d_blk_best_idx_) cudaFree(d_blk_best_idx_), d_blk_best_idx_ = nullptr;
        if (d_final_best_val_) cudaFree(d_final_best_val_), d_final_best_val_ = nullptr;
        if (d_final_best_idx_) cudaFree(d_final_best_idx_), d_final_best_idx_ = nullptr;

        cudaMalloc(reinterpret_cast<void**>(&d_A_), sizeof(float) * static_cast<size_t>(m_) * static_cast<size_t>(n_));
        cudaMalloc(reinterpret_cast<void**>(&d_x_), sizeof(float) * static_cast<size_t>(m_));
        cudaMalloc(reinterpret_cast<void**>(&d_scores_), sizeof(float) * static_cast<size_t>(n_));

        // Allocate per-block argmax buffers (TK only)
        #if defined(USE_TK)
        {
            constexpr int TN_TILE = 128;
            blk_count_ = (n_ + TN_TILE - 1) / TN_TILE;
            if (blk_count_ < 1) blk_count_ = 1;
            cudaMalloc(reinterpret_cast<void**>(&d_blk_best_abs_), sizeof(float) * static_cast<size_t>(blk_count_));
            cudaMalloc(reinterpret_cast<void**>(&d_blk_best_val_), sizeof(float) * static_cast<size_t>(blk_count_));
            cudaMalloc(reinterpret_cast<void**>(&d_blk_best_idx_), sizeof(int)   * static_cast<size_t>(blk_count_));
            cudaMalloc(reinterpret_cast<void**>(&d_final_best_val_), sizeof(float));
            cudaMalloc(reinterpret_cast<void**>(&d_final_best_idx_), sizeof(int));
        }
        #endif

        cudaMemcpyAsync(d_A_, dict_colmajor_f32_.data(), sizeof(float) * static_cast<size_t>(m_) * static_cast<size_t>(n_), cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);

        cuda_ready_ = true;
    } else {
        return;
    }
}

// Warmup: run a dummy kernel on zeros
template <typename Scalar>
void ACT_CUDA_TK_T<Scalar>::warmup_kernels() const {
    if constexpr (std::is_same_v<Scalar, float>) {
        if (!cuda_ready_) return;
        cudaMemsetAsync(d_x_, 0, sizeof(float) * static_cast<size_t>(m_), stream_);
        #if defined(USE_TK)
        {
            constexpr int TN_TILE = 128;
            int gridX = (n_ + TN_TILE - 1) / TN_TILE;
            dim3 grid(gridX, 1, 1);
            dim3 block(TN_TILE, 1, 1);
            gemv_tk_kernel<<<grid, block, 0, stream_>>>(
                d_A_, d_x_, d_scores_, m_, n_,
                d_blk_best_abs_, d_blk_best_val_, d_blk_best_idx_);
        }
        #else
        {
            const int MAX_BLOCKS_X = 65535;
            int blocksX = (n_ < MAX_BLOCKS_X) ? n_ : MAX_BLOCKS_X;
            int blocksY = (n_ + MAX_BLOCKS_X - 1) / MAX_BLOCKS_X;
            dim3 grid(blocksX > 0 ? blocksX : 1, blocksY > 0 ? blocksY : 1, 1);
            dim3 block(256, 1, 1);
            gemv_cols_kernel<<<grid, block, 0, stream_>>>(d_A_, d_x_, d_scores_, m_, n_, MAX_BLOCKS_X);
        }
        #endif
        cudaStreamSynchronize(stream_);
    } else {
        return;
    }
}

// Cleanup all CUDA resources
template <typename Scalar>
void ACT_CUDA_TK_T<Scalar>::cleanup_cuda() const {
    if (d_A_) { cudaFree(d_A_); d_A_ = nullptr; }
    if (d_x_) { cudaFree(d_x_); d_x_ = nullptr; }
    if (d_scores_) { cudaFree(d_scores_); d_scores_ = nullptr; }
    if (d_blk_best_abs_) { cudaFree(d_blk_best_abs_); d_blk_best_abs_ = nullptr; }
    if (d_blk_best_val_) { cudaFree(d_blk_best_val_); d_blk_best_val_ = nullptr; }
    if (d_blk_best_idx_) { cudaFree(d_blk_best_idx_); d_blk_best_idx_ = nullptr; }
    if (d_final_best_val_) { cudaFree(d_final_best_val_); d_final_best_val_ = nullptr; }
    if (d_final_best_idx_) { cudaFree(d_final_best_idx_); d_final_best_idx_ = nullptr; }
    blk_count_ = 0;
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    cuda_ready_ = false;
}

#endif // USE_CUDA

// Explicit instantiation
template class ACT_CUDA_TK_T<double>;
template class ACT_CUDA_TK_T<float>;
