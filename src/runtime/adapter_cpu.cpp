#include "adapter_cpu.h"

#include <cassert>
#include "../lir/module_utils.h"
#include <cstring>
#include <unordered_map>
#include <vector>

#include "ops_cpu.h"  // mini-llm-engine/src/ops_cpu.h

using namespace tsy;
using namespace tsy::lir;

namespace tsy::runtime {

namespace {

int64_t innermost(const Tensor& t) {
    return t.dims.empty() ? 0 : t.dims.back();
}

int64_t outerProduct(const Tensor& t) {
    int64_t o = 1;
    for (size_t i = 0; i + 1 < t.dims.size(); ++i) o *= t.dims[i];
    return o;
}

// Transpose a 2D row-major matrix [K, N] -> [N, K] into `out`.
// The adapter leans on this every matmul call; worst case it's one
// materialisation per invocation, which is acceptable for a W6 CPU
// reference path and lets callers keep canonical layouts.
void transpose2d(const float* src, float* dst, int64_t K, int64_t N) {
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t n = 0; n < N; ++n) {
            dst[n * K + k] = src[k * N + n];
        }
    }
}

// Cache of ones-vectors indexed by length, used as the gain input to
// rms_norm_cpu. rms_norm_cpu reads this pointer on every call and we
// don't want to reallocate the vector per-invocation.
const float* onesVector(int64_t n) {
    static std::unordered_map<int64_t, std::vector<float>> cache;
    auto& v = cache[n];
    if (static_cast<int64_t>(v.size()) != n) v.assign(n, 1.0f);
    return v.data();
}

}  // namespace

// ---------------------------------------------------------------------------
// primitive adapters
// ---------------------------------------------------------------------------

void adapterMatMul(const Tensor& a, const Tensor& b, Tensor& c) {
    // Assumes verifier has already asserted ranks ≥ 2 and matching inner
    // dims. For W6 we only model rank==2; higher-rank matmul is a later
    // week (it lowers to a batched loop around the 2D kernel).
    assert(a.dims.size() == 2 && b.dims.size() == 2 && c.dims.size() == 2);
    const int64_t M = a.dims[0];
    const int64_t K = a.dims[1];
    const int64_t N = b.dims[1];
    assert(b.dims[0] == K);
    assert(c.dims[0] == M && c.dims[1] == N);

    std::vector<float> bT(static_cast<size_t>(N) * static_cast<size_t>(K));
    transpose2d(b.data.data(), bT.data(), K, N);

    c.data.assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    matmul_cpu(a.data.data(), bT.data(), c.data.data(),
               static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
}

void adapterAdd(const Tensor& a, const Tensor& b, Tensor& c) {
    assert(a.data.size() == b.data.size());
    c.data.assign(a.data.size(), 0.0f);
    const size_t n = a.data.size();
    for (size_t i = 0; i < n; ++i) c.data[i] = a.data[i] + b.data[i];
}

void adapterSoftmax(const Tensor& x, Tensor& y) {
    const int64_t inner = innermost(x);
    const int64_t outer = outerProduct(x);
    y.data.assign(x.data.size(), 0.0f);
    // Copy X into Y first because softmax_inplace mutates its input.
    std::memcpy(y.data.data(), x.data.data(), x.data.size() * sizeof(float));
    for (int64_t r = 0; r < outer; ++r) {
        softmax_inplace(y.data.data() + r * inner, static_cast<int>(inner));
    }
}

void adapterRMSNorm(const Tensor& x, Tensor& y) {
    constexpr float kEps = 1e-6f;
    const int64_t inner = innermost(x);
    const int64_t outer = outerProduct(x);
    y.data.assign(x.data.size(), 0.0f);
    const float* w = onesVector(inner);
    rms_norm_cpu(x.data.data(), w, y.data.data(),
                 static_cast<int>(outer), static_cast<int>(inner), kEps);
}

void adapterTranspose(const Tensor& x, Tensor& c) {
    assert(x.dims.size() == 2);
    const int64_t M = x.dims[0];
    const int64_t N = x.dims[1];
    c.data.assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            c.data[j * M + i] = x.data[i * N + j];
        }
    }
}

void adapterReLU(const Tensor& x, Tensor& c) {
    const size_t n = x.data.size();
    c.data.assign(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float v = x.data[i];
        c.data[i] = v < 0.0f ? 0.0f : v;
    }
}

// ---------------------------------------------------------------------------
// executor: parallel to lir::runFunctionImpl but dispatches through adapters
// ---------------------------------------------------------------------------

namespace {

RunResult runFunctionAdapter(const Function& f, DiagnosticEngine& diag) {
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
            diag.error(s.loc, "cpu-adapter: call has no result buffer");
            r.ok = false;
            continue;
        }
        auto& out = r.buffers[s.result_buf];

        if (s.primitive == "matmul") {
            if (s.operand_bufs.size() != 2) {
                diag.error(s.loc, "cpu-adapter matmul: expected 2 operands");
                r.ok = false; continue;
            }
            adapterMatMul(r.buffers[s.operand_bufs[0]],
                          r.buffers[s.operand_bufs[1]], out);
        } else if (s.primitive == "add") {
            if (s.operand_bufs.size() != 2) {
                diag.error(s.loc, "cpu-adapter add: expected 2 operands");
                r.ok = false; continue;
            }
            adapterAdd(r.buffers[s.operand_bufs[0]],
                       r.buffers[s.operand_bufs[1]], out);
        } else if (s.primitive == "softmax") {
            if (s.operand_bufs.size() != 1) {
                diag.error(s.loc, "cpu-adapter softmax: expected 1 operand");
                r.ok = false; continue;
            }
            adapterSoftmax(r.buffers[s.operand_bufs[0]], out);
        } else if (s.primitive == "rmsnorm") {
            if (s.operand_bufs.size() != 1) {
                diag.error(s.loc, "cpu-adapter rmsnorm: expected 1 operand");
                r.ok = false; continue;
            }
            adapterRMSNorm(r.buffers[s.operand_bufs[0]], out);
        } else if (s.primitive == "transpose") {
            if (s.operand_bufs.size() != 1) {
                diag.error(s.loc, "cpu-adapter transpose: expected 1 operand");
                r.ok = false; continue;
            }
            adapterTranspose(r.buffers[s.operand_bufs[0]], out);
        } else if (s.primitive == "relu") {
            if (s.operand_bufs.size() != 1) {
                diag.error(s.loc, "cpu-adapter relu: expected 1 operand");
                r.ok = false; continue;
            }
            adapterReLU(r.buffers[s.operand_bufs[0]], out);
        } else {
            diag.error(s.loc, "cpu-adapter: unsupported primitive '" +
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

RunResult runWithCpuAdapter(const Module& m, DiagnosticEngine& diag) {
    const Function* f = tsy::lir::pickFirstTensorFunction(m);
    if (!f) {
        diag.error({}, "cpu-adapter: module has no runnable function");
        RunResult r; r.ok = false; return r;
    }
    return runFunctionAdapter(*f, diag);
}

RunResult runNamedWithCpuAdapter(const Module& m, const std::string& name,
                                 DiagnosticEngine& diag) {
    for (const auto& f : m.funcs) {
        if (f->name == name) return runFunctionAdapter(*f, diag);
    }
    diag.error({}, "cpu-adapter: function '" + name + "' not found");
    RunResult r; r.ok = false; return r;
}

}  // namespace tsy::runtime
