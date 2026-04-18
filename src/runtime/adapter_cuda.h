#pragma once

#include <string>

#include "../frontend/diagnostics.h"
#include "../lir/interpreter.h"   // NamedTensor / RunResult
#include "../lir/ir.h"

// CUDA runtime adapter — parallel to adapter_cpu.h. Self-contained FP32
// implementations using cuBLAS for GEMM and hand-written CUDA kernels
// for add/softmax/rmsnorm. Each adapter* entry point manages its own
// host<->device staging and allocations; callers pass plain host-side
// NamedTensor and receive host-side NamedTensor back.
//
// Layout + semantics match adapter_cpu.h verbatim. Tolerance when
// comparing outputs against adapter_cpu: atol=1e-4, rtol=1e-3.

namespace tsy::runtime {

using Tensor = tsy::lir::NamedTensor;

// MatMul: C[M,N] = A[M,K] @ B[K,N], FP32, row-major.
// Internally: cuBLAS sgemm via the col-major trick
//   row-major (A @ B) == col-major (B^T^T @ A^T) == col-major (B @ A^T)
void adapterMatMulCuda(const Tensor& a, const Tensor& b, Tensor& c);

// Elementwise add, same shape required.
void adapterAddCuda(const Tensor& a, const Tensor& b, Tensor& c);

// Softmax along the innermost dim. Numerically-stable two-pass
// max/sum/normalize kernel, one block per outer row.
void adapterSoftmaxCuda(const Tensor& x, Tensor& y);

// RMSNorm along the innermost dim with eps=1e-6 and an implicit
// ones-weight vector (the HIR doesn't expose gain yet — matches CPU
// adapter behaviour). Warp-shuffle reduce over x^2.
void adapterRMSNormCuda(const Tensor& x, Tensor& y);

// Executor entry points (parallel to runWithCpuAdapter).
tsy::lir::RunResult runWithCudaAdapter(const tsy::lir::Module& m,
                                       tsy::DiagnosticEngine& diag);

tsy::lir::RunResult runNamedWithCudaAdapter(const tsy::lir::Module& m,
                                            const std::string& name,
                                            tsy::DiagnosticEngine& diag);

}  // namespace tsy::runtime
