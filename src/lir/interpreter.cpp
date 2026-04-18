#include "interpreter.h"
#include "module_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <unordered_map>

using namespace tsy;

namespace tsy::lir {

namespace {

// --- deterministic input ----------------------------------------------------

float detValue(int buf_idx, int64_t elem_idx) {
    // Linear fill that's easy to hand-verify: each new buffer shifts by 0.5,
    // each new element steps by 0.1. Keeps values within a friendly fp32
    // range for 32-element test tensors.
    return static_cast<float>(buf_idx) * 0.5f + static_cast<float>(elem_idx) * 0.1f;
}

// --- primitive kernels ------------------------------------------------------

void kernelMatMul(const NamedTensor& a, const NamedTensor& b, NamedTensor& c) {
    // Expect a:[M,K], b:[K,N], c:[M,N]. Shape correctness is already the
    // verifier's job; we trust it here.
    int64_t M = a.dims[0];
    int64_t K = a.dims[1];
    int64_t N = b.dims[1];
    c.data.assign(M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float aik = a.data[i * K + k];
            for (int64_t j = 0; j < N; ++j) {
                c.data[i * N + j] += aik * b.data[k * N + j];
            }
        }
    }
}

void kernelAdd(const NamedTensor& a, const NamedTensor& b, NamedTensor& c) {
    const int64_t n = static_cast<int64_t>(a.data.size());
    c.data.assign(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) c.data[i] = a.data[i] + b.data[i];
}

// Softmax along the innermost dim.
void kernelSoftmax(const NamedTensor& x, NamedTensor& y) {
    int64_t inner = x.dims.back();
    int64_t outer = 1;
    for (size_t i = 0; i + 1 < x.dims.size(); ++i) outer *= x.dims[i];
    y.data.assign(x.data.size(), 0.0f);
    for (int64_t r = 0; r < outer; ++r) {
        const float* row = x.data.data() + r * inner;
        float* out = y.data.data() + r * inner;
        float mx = row[0];
        for (int64_t i = 1; i < inner; ++i) mx = std::max(mx, row[i]);
        float sum = 0.0f;
        for (int64_t i = 0; i < inner; ++i) {
            out[i] = std::exp(row[i] - mx);
            sum += out[i];
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int64_t i = 0; i < inner; ++i) out[i] *= inv;
    }
}

// RMSNorm along the innermost dim with eps=1e-6. Gain vector is implicit 1.
void kernelRMSNorm(const NamedTensor& x, NamedTensor& y) {
    constexpr float kEps = 1e-6f;
    int64_t inner = x.dims.back();
    int64_t outer = 1;
    for (size_t i = 0; i + 1 < x.dims.size(); ++i) outer *= x.dims[i];
    y.data.assign(x.data.size(), 0.0f);
    for (int64_t r = 0; r < outer; ++r) {
        const float* row = x.data.data() + r * inner;
        float* out = y.data.data() + r * inner;
        float sq = 0.0f;
        for (int64_t i = 0; i < inner; ++i) sq += row[i] * row[i];
        float rms = std::sqrt(sq / static_cast<float>(inner) + kEps);
        float inv = 1.0f / rms;
        for (int64_t i = 0; i < inner; ++i) out[i] = row[i] * inv;
    }
}

// 2-D transpose: y[j, i] = x[i, j]. Caller ensures x is rank 2 (verifier's job).
void kernelTranspose(const NamedTensor& x, NamedTensor& y) {
    int64_t M = x.dims[0];
    int64_t N = x.dims[1];
    y.data.assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            y.data[j * M + i] = x.data[i * N + j];
        }
    }
}

// Elementwise ReLU: y[i] = max(0, x[i]). Shape preserved.
void kernelReLU(const NamedTensor& x, NamedTensor& y) {
    const int64_t n = static_cast<int64_t>(x.data.size());
    y.data.assign(n, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        float v = x.data[i];
        y.data[i] = v < 0.0f ? 0.0f : v;
    }
}

// --- executor ---------------------------------------------------------------

NamedTensor materialise(const Buffer& b) {
    NamedTensor t;
    t.name = b.name;
    t.dims = b.dims;
    t.data.assign(b.numElements(), 0.0f);
    return t;
}

RunResult runFunctionImpl(const Function& f, DiagnosticEngine& diag) {
    RunResult r;
    r.function_name = f.name;
    r.buffers.reserve(f.buffers.size());
    for (const auto& b : f.buffers) r.buffers.push_back(materialise(b));

    // Fill parameter buffers with deterministic inputs.
    for (size_t i = 0; i < f.params.size(); ++i) {
        fillDeterministic(r.buffers[f.params[i]], static_cast<int>(i));
    }

    for (const auto& s : f.body) {
        if (s.kind == StmtKind::Return) break;
        if (s.kind != StmtKind::Call) continue;

        if (s.result_buf < 0) {
            diag.error(s.loc, "interpreter: call has no result buffer");
            r.ok = false;
            continue;
        }
        auto& out = r.buffers[s.result_buf];

        if (s.primitive == "matmul") {
            if (s.operand_bufs.size() != 2) { diag.error(s.loc, "matmul: expected 2 operands"); r.ok = false; continue; }
            kernelMatMul(r.buffers[s.operand_bufs[0]], r.buffers[s.operand_bufs[1]], out);
        } else if (s.primitive == "add") {
            if (s.operand_bufs.size() != 2) { diag.error(s.loc, "add: expected 2 operands"); r.ok = false; continue; }
            kernelAdd(r.buffers[s.operand_bufs[0]], r.buffers[s.operand_bufs[1]], out);
        } else if (s.primitive == "softmax") {
            if (s.operand_bufs.size() != 1) { diag.error(s.loc, "softmax: expected 1 operand"); r.ok = false; continue; }
            kernelSoftmax(r.buffers[s.operand_bufs[0]], out);
        } else if (s.primitive == "rmsnorm") {
            if (s.operand_bufs.size() != 1) { diag.error(s.loc, "rmsnorm: expected 1 operand"); r.ok = false; continue; }
            kernelRMSNorm(r.buffers[s.operand_bufs[0]], out);
        } else if (s.primitive == "transpose") {
            if (s.operand_bufs.size() != 1) { diag.error(s.loc, "transpose: expected 1 operand"); r.ok = false; continue; }
            kernelTranspose(r.buffers[s.operand_bufs[0]], out);
        } else if (s.primitive == "relu") {
            if (s.operand_bufs.size() != 1) { diag.error(s.loc, "relu: expected 1 operand"); r.ok = false; continue; }
            kernelReLU(r.buffers[s.operand_bufs[0]], out);
        } else {
            diag.error(s.loc, "interpreter: unknown primitive '" + s.primitive + "'");
            r.ok = false;
        }
    }

    for (size_t i = 0; i < f.params.size(); ++i) {
        r.buffers[f.params[i]].is_param = true;
    }
    return r;
}

}  // namespace

float deterministicValue(int buf_idx, int64_t elem_idx) { return detValue(buf_idx, elem_idx); }

void fillDeterministic(NamedTensor& t, int buf_idx) {
    for (int64_t i = 0; i < static_cast<int64_t>(t.data.size()); ++i) {
        t.data[i] = detValue(buf_idx, i);
    }
}

RunResult runFirstTensorFunction(const Module& m, DiagnosticEngine& diag) {
    const Function* f = pickFirstTensorFunction(m);
    if (!f) {
        diag.error({}, "interpreter: module has no runnable function");
        RunResult r;
        r.ok = false;
        return r;
    }
    return runFunctionImpl(*f, diag);
}

RunResult runNamedFunction(const Module& m, const std::string& name,
                           DiagnosticEngine& diag) {
    for (const auto& f : m.funcs) {
        if (f->name == name) return runFunctionImpl(*f, diag);
    }
    diag.error({}, "interpreter: function '" + name + "' not found");
    RunResult r;
    r.ok = false;
    return r;
}

// Readable dump: shape header, then row-major data wrapped on the innermost
// dim. Keep precision short enough that integration tests can match it with
// regexes and hand calculations.
void printRunResult(std::ostream& os, const RunResult& r) {
    os << "function: " << r.function_name << "\n";
    for (const auto& t : r.buffers) {
        os << (t.is_param ? "  input "  : "  local ") << t.name << " shape=[";
        for (size_t i = 0; i < t.dims.size(); ++i) {
            if (i) os << ",";
            os << t.dims[i];
        }
        os << "]:\n";
        int64_t inner = t.dims.empty() ? 0 : t.dims.back();
        if (inner == 0) { os << "    (scalar)\n"; continue; }
        int64_t outer = static_cast<int64_t>(t.data.size()) / std::max<int64_t>(inner, 1);
        char buf[32];
        for (int64_t r = 0; r < outer; ++r) {
            os << "   ";
            for (int64_t j = 0; j < inner; ++j) {
                std::snprintf(buf, sizeof(buf), " %8.4f", t.data[r * inner + j]);
                os << buf;
            }
            os << "\n";
        }
    }
}

}  // namespace tsy::lir
