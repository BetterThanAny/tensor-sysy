// W9 ScheduleCudaPass — shape-based kernel variant picker for matmul.
//
// The picker rules (initial thresholds, tuned after tsy-bench runs):
//   M*N < 1024                              → "naive"
//   aligned && M,N,K >= 128 && M*N <= 256^2 → "tiled"
//   otherwise                               → "cublas"
// where aligned == (M%128==0 && N%128==0 && K%8==0).
//
// Only matmul calls are scheduled. Other primitives (add/softmax/rmsnorm)
// have a single CUDA kernel and need no variant attr.

#include "pass_manager.h"

#include <string>

namespace tsy::passes {

namespace {

std::string pickMatmulVariant(int64_t M, int64_t K, int64_t N) {
    if (M * N < 1024) return "naive";
    const bool aligned =
        (M % 128 == 0) && (N % 128 == 0) && (K % 8 == 0);
    const bool large_enough =
        (M >= 128) && (N >= 128) && (K >= 128);
    if (aligned && large_enough && M * N <= 256 * 256) return "tiled";
    return "cublas";
}

}  // namespace

void runScheduleCuda(tsy::lir::Module& m, tsy::DiagnosticEngine& /*diag*/) {
    for (auto& f : m.funcs) {
        for (auto& s : f->body) {
            if (s.kind != tsy::lir::StmtKind::Call) continue;
            if (s.primitive != "matmul") continue;
            if (s.operand_bufs.size() != 2 || s.result_buf < 0) continue;
            const auto& A = f->buffers[s.operand_bufs[0]];
            const auto& B = f->buffers[s.operand_bufs[1]];
            if (A.dims.size() != 2 || B.dims.size() != 2) continue;
            const int64_t M = A.dims[0];
            const int64_t K = A.dims[1];
            const int64_t N = B.dims[1];
            s.attrs["variant"] = pickMatmulVariant(M, K, N);
        }
    }
}

}  // namespace tsy::passes
