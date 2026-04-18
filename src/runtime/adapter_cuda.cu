#include "adapter_cuda.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace tsy;
using namespace tsy::lir;

namespace tsy::runtime {

namespace {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(_err));        \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _st = (call);                                           \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",         \
                         __FILE__, __LINE__, static_cast<int>(_st));           \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Lazy-initialised, process-singleton cuBLAS handle. No locking because the
// executor is single-threaded.
cublasHandle_t getCublasHandle() {
    static cublasHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, []() { CUBLAS_CHECK(cublasCreate(&handle)); });
    return handle;
}

// Device-side ones-weight cache keyed by inner dim. No locking because the
// executor is single-threaded; the cache leaks on exit by design (same
// rationale as the cuBLAS handle).
const float* onesDeviceVector(int64_t inner) {
    static std::unordered_map<int64_t, float*> cache;
    auto it = cache.find(inner);
    if (it != cache.end()) return it->second;
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, inner * sizeof(float)));
    std::vector<float> host(inner, 1.0f);
    CUDA_CHECK(cudaMemcpy(d, host.data(), inner * sizeof(float),
                          cudaMemcpyHostToDevice));
    cache[inner] = d;
    return d;
}

}  // namespace

__global__ void addKernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// C[M,N] = A[M,K] @ B[K,N] (row-major, FP32).
// cuBLAS is col-major; use the identity
//   row-major (A @ B) == col-major (B @ A)  (since col-major M is row-major M^T)
// so call sgemm(OP_N, OP_N, N, M, K, B, A, C).
void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c) {
    assert(a.dims.size() == 2 && b.dims.size() == 2 && c.dims.size() == 2);
    const int M = static_cast<int>(a.dims[0]);
    const int K = static_cast<int>(a.dims[1]);
    const int N = static_cast<int>(b.dims[1]);
    assert(b.dims[0] == K);
    assert(c.dims[0] == M && c.dims[1] == N);

    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, a.data.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data.data(), bytesB, cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(
        getCublasHandle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB, N,       // leading dim of B in col-major view is N
        dA, K,       // leading dim of A in col-major view is K
        &beta,
        dC, N));

    c.data.assign(static_cast<size_t>(M) * N, 0.0f);
    CUDA_CHECK(cudaMemcpy(c.data.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

void adapterAddCuda(const Tensor& a, const Tensor& b, Tensor& c) {
    assert(a.data.size() == b.data.size());
    const int n = static_cast<int>(a.data.size());
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, a.data.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b.data.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid  = (n + block - 1) / block;
    addKernel<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());

    c.data.assign(static_cast<size_t>(n), 0.0f);
    CUDA_CHECK(cudaMemcpy(c.data.data(), dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}
__global__ void softmaxRowKernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int inner) {
    extern __shared__ float smem[];   // blockDim.x floats
    const int row = blockIdx.x;
    const float* xr = x + row * inner;
    float*       yr = y + row * inner;

    // Pass 1: max reduction
    float tmax = -1e30f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        tmax = fmaxf(tmax, xr[i]);
    }
    smem[threadIdx.x] = tmax;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x],
                                       smem[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float row_max = smem[0];
    __syncthreads();

    // Pass 2: exp + sum reduction
    float tsum = 0.0f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float v = expf(xr[i] - row_max);
        yr[i] = v;
        tsum += v;
    }
    smem[threadIdx.x] = tsum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / smem[0];
    __syncthreads();

    // Pass 3: normalise
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        yr[i] *= inv;
    }
}

void adapterSoftmaxCuda(const Tensor& x, Tensor& y) {
    assert(!x.dims.empty());
    const int64_t inner = x.dims.back();
    int64_t outer = 1;
    for (size_t i = 0; i + 1 < x.dims.size(); ++i) outer *= x.dims[i];
    const size_t n = x.data.size();
    const size_t bytes = n * sizeof(float);

    float *dX = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dY, bytes));
    CUDA_CHECK(cudaMemcpy(dX, x.data.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const size_t smem_bytes = static_cast<size_t>(block) * sizeof(float);
    softmaxRowKernel<<<static_cast<int>(outer), block, smem_bytes>>>(
        dX, dY, static_cast<int>(inner));
    CUDA_CHECK(cudaGetLastError());

    y.data.assign(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(y.data.data(), dY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
}
__global__ void rmsnormRowKernel(const float* __restrict__ x,
                                  const float* __restrict__ w,
                                  float* __restrict__ y,
                                  int inner, float eps) {
    extern __shared__ float smem[];   // 1 float per warp
    const int row = blockIdx.x;
    const float* xr = x + row * inner;
    float*       yr = y + row * inner;

    // Phase 1: sum(x^2) in FP32
    float sum = 0.0f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    // Intra-warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    const int warp_id  = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    const int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) smem[0] = sum;
    }
    __syncthreads();

    const float rms_inv = rsqrtf(smem[0] / static_cast<float>(inner) + eps);

    // Phase 2: normalise + scale
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        yr[i] = xr[i] * rms_inv * w[i];
    }
}

void adapterRMSNormCuda(const Tensor& x, Tensor& y) {
    constexpr float kEps = 1e-6f;
    assert(!x.dims.empty());
    const int64_t inner = x.dims.back();
    int64_t outer = 1;
    for (size_t i = 0; i + 1 < x.dims.size(); ++i) outer *= x.dims[i];
    const size_t n = x.data.size();
    const size_t bytes = n * sizeof(float);

    float *dX = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dY, bytes));
    CUDA_CHECK(cudaMemcpy(dX, x.data.data(), bytes, cudaMemcpyHostToDevice));
    const float* dW = onesDeviceVector(inner);

    const int block     = 128;  // 4 warps
    const int num_warps = block / 32;
    const size_t smem_bytes = static_cast<size_t>(num_warps) * sizeof(float);
    rmsnormRowKernel<<<static_cast<int>(outer), block, smem_bytes>>>(
        dX, dW, dY, static_cast<int>(inner), kEps);
    CUDA_CHECK(cudaGetLastError());

    y.data.assign(n, 0.0f);
    CUDA_CHECK(cudaMemcpy(y.data.data(), dY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
}

// Executor added in Task 10.
tsy::lir::RunResult runWithCudaAdapter(const tsy::lir::Module&,
                                       tsy::DiagnosticEngine& diag) {
    diag.error({}, "cuda-adapter executor not implemented (Task 10)");
    RunResult r; r.ok = false; return r;
}
tsy::lir::RunResult runNamedWithCudaAdapter(const tsy::lir::Module&,
                                            const std::string&,
                                            tsy::DiagnosticEngine& diag) {
    diag.error({}, "cuda-adapter executor not implemented (Task 10)");
    RunResult r; r.ok = false; return r;
}

}  // namespace tsy::runtime
