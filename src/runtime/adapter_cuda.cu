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

}  // namespace

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

// Other primitives added in later tasks — linker-time stubs until then.
void adapterAddCuda(const Tensor&, const Tensor&, Tensor&) {
    std::fprintf(stderr, "adapterAddCuda: not implemented (Task 5)\n");
    std::abort();
}
void adapterSoftmaxCuda(const Tensor&, Tensor&) {
    std::fprintf(stderr, "adapterSoftmaxCuda: not implemented (Task 6)\n");
    std::abort();
}
void adapterRMSNormCuda(const Tensor&, Tensor&) {
    std::fprintf(stderr, "adapterRMSNormCuda: not implemented (Task 7)\n");
    std::abort();
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
