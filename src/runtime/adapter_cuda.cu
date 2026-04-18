#include "adapter_cuda.h"

#include <cassert>
#include "../lir/module_utils.h"
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

// One thread computes one output element via a dot-product over K.
// Launch: block(32,32), grid(ceil(N/32), ceil(M/32)).
__global__ void matmulNaiveKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// Register-tiled GEMM — 128x128 output tile per block, 8x8 register tile
// per thread, BK=8 k-strip. Ported from mini-llm-engine/cuda-kernels/gemm/
// gemm.cu (gemm_register_tiled). Requires M%128==0 && N%128==0 && K%8==0.
// Launch: block(16,16)=256 threads, grid(N/128, M/128).
#define TSY_BM 128
#define TSY_BN 128
#define TSY_BK 8
#define TSY_TM 8
#define TSY_TN 8
__global__ void matmulTiledKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * (TSY_BN / TSY_TN) + tx;

    const int bm = blockIdx.y * TSY_BM;
    const int bn = blockIdx.x * TSY_BN;

    __shared__ float As[TSY_BM][TSY_BK + 1];
    __shared__ float Bs[TSY_BK][TSY_BN + 4];

    float C_reg[TSY_TM][TSY_TN];
    #pragma unroll
    for (int i = 0; i < TSY_TM; i++)
        #pragma unroll
        for (int j = 0; j < TSY_TN; j++)
            C_reg[i][j] = 0.f;

    const int a_row  = tid / 2;
    const int a_col4 = (tid % 2) * 4;
    const int b_row  = tid / (TSY_BN / 4);
    const int b_col4 = (tid % (TSY_BN / 4)) * 4;

    for (int k_tile = 0; k_tile < K; k_tile += TSY_BK) {
        {
            const int gm = bm + a_row;
            const int gk = k_tile + a_col4;
            if (gm < M && gk + 3 < K) {
                float4 a4 = *reinterpret_cast<const float4*>(A + gm * K + gk);
                As[a_row][a_col4 + 0] = a4.x;
                As[a_row][a_col4 + 1] = a4.y;
                As[a_row][a_col4 + 2] = a4.z;
                As[a_row][a_col4 + 3] = a4.w;
            } else {
                #pragma unroll
                for (int dk = 0; dk < 4; dk++) {
                    int k = gk + dk, m = gm;
                    As[a_row][a_col4 + dk] = (m < M && k < K) ? A[m * K + k] : 0.f;
                }
            }
        }
        {
            const int gk = k_tile + b_row;
            const int gn = bn + b_col4;
            if (gk < K && gn + 3 < N) {
                float4 b4 = *reinterpret_cast<const float4*>(B + gk * N + gn);
                Bs[b_row][b_col4 + 0] = b4.x;
                Bs[b_row][b_col4 + 1] = b4.y;
                Bs[b_row][b_col4 + 2] = b4.z;
                Bs[b_row][b_col4 + 3] = b4.w;
            } else {
                #pragma unroll
                for (int dn = 0; dn < 4; dn++) {
                    int k = gk, n = gn + dn;
                    Bs[b_row][b_col4 + dn] = (k < K && n < N) ? B[k * N + n] : 0.f;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TSY_BK; k++) {
            float a_reg[TSY_TM], b_reg[TSY_TN];
            #pragma unroll
            for (int tm = 0; tm < TSY_TM; tm++)
                a_reg[tm] = As[ty * TSY_TM + tm][k];
            #pragma unroll
            for (int tn = 0; tn < TSY_TN; tn++)
                b_reg[tn] = Bs[k][tx * TSY_TN + tn];
            #pragma unroll
            for (int tm = 0; tm < TSY_TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < TSY_TN; tn++)
                    C_reg[tm][tn] += a_reg[tm] * b_reg[tn];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < TSY_TM; tm++) {
        const int gm = bm + ty * TSY_TM + tm;
        if (gm >= M) continue;
        {
            const int gn = bn + tx * TSY_TN;
            if (gn + 3 < N) {
                float4 c4 = {C_reg[tm][0], C_reg[tm][1],
                             C_reg[tm][2], C_reg[tm][3]};
                *reinterpret_cast<float4*>(C + gm * N + gn) = c4;
            } else {
                for (int i = 0; i < 4 && gn + i < N; i++)
                    C[gm * N + gn + i] = C_reg[tm][i];
            }
        }
        {
            const int gn = bn + tx * TSY_TN + 4;
            if (gn + 3 < N) {
                float4 c4 = {C_reg[tm][4], C_reg[tm][5],
                             C_reg[tm][6], C_reg[tm][7]};
                *reinterpret_cast<float4*>(C + gm * N + gn) = c4;
            } else {
                for (int i = 0; i < 4 && gn + i < N; i++)
                    C[gm * N + gn + i] = C_reg[tm][4 + i];
            }
        }
    }
}
#undef TSY_BM
#undef TSY_BN
#undef TSY_BK
#undef TSY_TM
#undef TSY_TN

__global__ void addKernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c,
                       const std::string& variant) {
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

    if (variant == "naive") {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32);
        matmulNaiveKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    } else if (variant == "tiled") {
        assert(M % 128 == 0 && N % 128 == 0 && K % 8 == 0 &&
               "tiled variant requires M%128==0 && N%128==0 && K%8==0");
        dim3 block(16, 16);
        dim3 grid(N / 128, M / 128);
        matmulTiledKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // "" or "cublas": W8 path.
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            getCublasHandle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, N,
            dA, K,
            &beta,
            dC, N));
    }

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

namespace {

RunResult runFunctionCudaAdapter(const Function& f, DiagnosticEngine& diag) {
    RunResult r;
    r.function_name = f.name;
    r.buffers.reserve(f.buffers.size());
    for (const auto& b : f.buffers) {
        NamedTensor t;
        t.name = b.name;
        t.dims = b.dims;
        t.data.assign(b.numElements(), 0.0f);
        r.buffers.push_back(std::move(t));
    }

    for (size_t i = 0; i < f.params.size(); ++i) {
        fillDeterministic(r.buffers[f.params[i]], static_cast<int>(i));
    }

    for (const auto& s : f.body) {
        if (s.kind == StmtKind::Return) break;
        if (s.kind != StmtKind::Call) continue;
        if (s.result_buf < 0) {
            diag.error(s.loc, "cuda-adapter: call has no result buffer");
            r.ok = false;
            continue;
        }
        auto& out = r.buffers[s.result_buf];

        if (s.primitive == "matmul") {
            if (s.operand_bufs.size() != 2) {
                diag.error(s.loc, "cuda-adapter matmul: expected 2 operands");
                r.ok = false; continue;
            }
            std::string variant;
            auto it = s.attrs.find("variant");
            if (it != s.attrs.end()) variant = it->second;
            adapterMatMulCuda(r.buffers[s.operand_bufs[0]],
                              r.buffers[s.operand_bufs[1]], out, variant);
        } else if (s.primitive == "add") {
            if (s.operand_bufs.size() != 2) {
                diag.error(s.loc, "cuda-adapter add: expected 2 operands");
                r.ok = false; continue;
            }
            adapterAddCuda(r.buffers[s.operand_bufs[0]],
                           r.buffers[s.operand_bufs[1]], out);
        } else if (s.primitive == "softmax") {
            if (s.operand_bufs.size() != 1) {
                diag.error(s.loc, "cuda-adapter softmax: expected 1 operand");
                r.ok = false; continue;
            }
            adapterSoftmaxCuda(r.buffers[s.operand_bufs[0]], out);
        } else if (s.primitive == "rmsnorm") {
            if (s.operand_bufs.size() != 1) {
                diag.error(s.loc, "cuda-adapter rmsnorm: expected 1 operand");
                r.ok = false; continue;
            }
            adapterRMSNormCuda(r.buffers[s.operand_bufs[0]], out);
        } else {
            diag.error(s.loc, "cuda-adapter: unsupported primitive '" +
                                   s.primitive + "'");
            r.ok = false;
        }
    }

    for (size_t i = 0; i < f.params.size(); ++i) {
        r.buffers[f.params[i]].is_param = true;
    }
    return r;
}

}  // namespace

RunResult runWithCudaAdapter(const Module& m, DiagnosticEngine& diag) {
    const Function* f = tsy::lir::pickFirstTensorFunction(m);
    if (!f) {
        diag.error({}, "cuda-adapter: module has no runnable function");
        RunResult r; r.ok = false; return r;
    }
    return runFunctionCudaAdapter(*f, diag);
}

RunResult runNamedWithCudaAdapter(const Module& m, const std::string& name,
                                   DiagnosticEngine& diag) {
    for (const auto& f : m.funcs) {
        if (f->name == name) return runFunctionCudaAdapter(*f, diag);
    }
    diag.error({}, "cuda-adapter: function '" + name + "' not found");
    RunResult r; r.ok = false; return r;
}

}  // namespace tsy::runtime
