#pragma once

#include <string>

#include "../frontend/diagnostics.h"
#include "../lir/interpreter.h"   // NamedTensor / RunResult
#include "../lir/ir.h"

// CPU runtime adapter: the thin boundary layer that dispatches
// TensorSysY LIR operations onto the kernels shipped in
// `mini-llm-engine/src/ops_cpu.*`. Where mini-llm-engine's semantics
// differ from TensorSysY's HIR contract (most notably matmul layout,
// and rmsnorm's required gain vector), the adapter handles the
// conversion in-place so higher layers see a clean
// "input tensor -> output tensor" interface.
//
// What this adapter is NOT responsible for:
//   - Choosing a backend (the executor/CLI does that).
//   - Shape / dtype validation (the HIR verifier already enforced it).
//   - Backprop or training-mode behaviour; every op is forward-only.

namespace tsy::runtime {

using Tensor = tsy::lir::NamedTensor;

// MatMul with canonical TensorSysY semantics:
//   C[M, N] = A[M, K] @ B[K, N]
//
// mini-llm-engine/ops_cpu.matmul_cpu expects B pre-transposed to [N, K]
// because it's built for weight matrices stored as
// [out_features, in_features]. The adapter materialises that transpose
// on the fly so callers never need to know about the layout mismatch.
void adapterMatMul(const Tensor& a, const Tensor& b, Tensor& c);

// Elementwise add (C = A + B). No matching kernel upstream, so the
// adapter implements it directly. Lives here rather than in a grab-bag
// "utils" file so every CPU primitive has one home.
void adapterAdd(const Tensor& a, const Tensor& b, Tensor& c);

// Softmax along the innermost dim. Uses mini-llm-engine's
// `softmax_inplace` per outer row after copying X into Y; the upstream
// kernel mutates its input.
void adapterSoftmax(const Tensor& x, Tensor& y);

// RMSNorm along the innermost dim with eps=1e-6. ops_cpu.rms_norm_cpu
// requires a per-feature gain vector `w`; TensorSysY's HIR doesn't
// expose one (yet), so the adapter feeds in an ones-filled vector that
// is cached per distinct innermost-dim size across calls.
void adapterRMSNorm(const Tensor& x, Tensor& y);

// Execute a whole LIR Function via the CPU adapter. Parallel to
// lir::runFunctionImpl() but dispatches every Call through the
// adapter*() primitives above. Deterministic parameter fill is shared
// with the native interpreter so outputs can be compared element-wise.
tsy::lir::RunResult runWithCpuAdapter(const tsy::lir::Module& m,
                                      tsy::DiagnosticEngine& diag);

tsy::lir::RunResult runNamedWithCpuAdapter(const tsy::lir::Module& m,
                                           const std::string& name,
                                           tsy::DiagnosticEngine& diag);

}  // namespace tsy::runtime
