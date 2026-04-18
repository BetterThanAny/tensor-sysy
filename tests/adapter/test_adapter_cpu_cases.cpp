// W6 adapter_cpu cases.
//
// Exercises the runtime adapter on two axes:
//   1. Direct primitive tests with hand-computed references that call the
//      adapter functions on ad-hoc NamedTensor inputs. These pin down the
//      semantics specified in PLAN.md §W6 (matmul layout, softmax last
//      dim, rmsnorm last dim, add elementwise).
//   2. End-to-end interpreter-vs-adapter parity on every W4 run fixture:
//      both backends must agree within fp32 tolerance.

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/frontend/parser_driver.h"
#include "../../src/hir/lowering.h"
#include "../../src/hir/verifier.h"
#include "../../src/lir/interpreter.h"
#include "../../src/lir/lowering.h"
#include "../../src/runtime/adapter_cpu.h"

using tsy::lir::NamedTensor;
using tsy::runtime::adapterAdd;
using tsy::runtime::adapterMatMul;
using tsy::runtime::adapterRMSNorm;
using tsy::runtime::adapterSoftmax;

namespace {

int g_failures = 0;

void fail(const std::string& label, const std::string& why) {
    std::cerr << "FAIL[" << label << "]: " << why << "\n";
    ++g_failures;
}

bool approx(float a, float b, float atol) {
    return std::fabs(a - b) <= atol;
}

NamedTensor makeTensor(const std::string& name,
                       const std::vector<int64_t>& dims,
                       const std::vector<float>& data) {
    NamedTensor t;
    t.name = name;
    t.dims = dims;
    t.data = data;
    return t;
}

NamedTensor zeroTensor(const std::string& name,
                       const std::vector<int64_t>& dims) {
    NamedTensor t;
    t.name = name;
    t.dims = dims;
    int64_t n = 1;
    for (auto d : dims) n *= d;
    t.data.assign(n, 0.0f);
    return t;
}

bool checkTensor(const std::string& label, const NamedTensor& t,
                 const std::vector<float>& expected, float atol) {
    if (t.data.size() != expected.size()) {
        fail(label, "size mismatch: got " + std::to_string(t.data.size()) +
                        ", expected " + std::to_string(expected.size()));
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!approx(t.data[i], expected[i], atol)) {
            std::ostringstream oss;
            oss << "element " << i << " differs: got " << t.data[i]
                << ", expected " << expected[i] << " (atol=" << atol << ")";
            fail(label, oss.str());
            return false;
        }
    }
    return true;
}

// --- direct primitive tests ------------------------------------------------

// A[2,3] @ B[3,2] = C[2,2]. Non-symmetric shapes stress the transpose-on-the-fly
// inside adapterMatMul; a bug in the B^T materialisation would misalign
// output elements.
void caseAdapterCpuMatmulLayout() {
    const std::string label = "adapter_cpu_matmul_layout";
    auto A = makeTensor("A", {2, 3}, {1, 2, 3, 4, 5, 6});
    auto B = makeTensor("B", {3, 2}, {1, 0, 0, 1, 1, 1});
    auto C = zeroTensor("C", {2, 2});
    adapterMatMul(A, B, C);
    // A @ B:
    //   row0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    //   row1: [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
    if (!checkTensor(label, C, {4.0f, 5.0f, 10.0f, 11.0f}, 1e-5f)) return;
    std::cout << "PASS[" << label << "]\n";
}

// Softmax along last dim with two independent rows. Explicit asymmetric
// values (row 0 ascending, row 1 descending) catches a wrong-axis
// reduction bug.
void caseAdapterCpuSoftmaxDim() {
    const std::string label = "adapter_cpu_softmax_dim";
    auto X = makeTensor("X", {2, 3}, {1, 2, 3, 3, 2, 1});
    auto Y = zeroTensor("Y", {2, 3});
    adapterSoftmax(X, Y);
    // Row 0: exp(-2), exp(-1), exp(0) / sum ≈ [0.0900, 0.2447, 0.6652]
    // Row 1: mirrored.
    std::vector<float> expected = {
        0.09003f, 0.24473f, 0.66524f,
        0.66524f, 0.24473f, 0.09003f,
    };
    if (!checkTensor(label, Y, expected, 1e-4f)) return;
    // Each row must sum to 1.
    for (int r = 0; r < 2; ++r) {
        float s = 0.0f;
        for (int i = 0; i < 3; ++i) s += Y.data[r * 3 + i];
        if (!approx(s, 1.0f, 1e-4f)) {
            fail(label, "row " + std::to_string(r) + " sum=" + std::to_string(s));
            return;
        }
    }
    std::cout << "PASS[" << label << "]\n";
}

// RMSNorm along last dim with two rows; distinct magnitudes so a wrong
// reduction across rows would show up immediately.
void caseAdapterCpuRmsnormLastDim() {
    const std::string label = "adapter_cpu_rmsnorm_last_dim";
    auto X = makeTensor("X", {2, 3}, {1, 2, 3, 4, 5, 6});
    auto Y = zeroTensor("Y", {2, 3});
    adapterRMSNorm(X, Y);
    // Row 0: rms = sqrt(14/3 + 1e-6) ≈ 2.16025;  Y = X / rms
    // Row 1: rms = sqrt(77/3 + 1e-6) ≈ 5.06623
    std::vector<float> expected = {
        1.0f / 2.16025f, 2.0f / 2.16025f, 3.0f / 2.16025f,
        4.0f / 5.06623f, 5.0f / 5.06623f, 6.0f / 5.06623f,
    };
    if (!checkTensor(label, Y, expected, 1e-3f)) return;
    std::cout << "PASS[" << label << "]\n";
}

// Elementwise add; sanity plus picks up any off-by-one in the loop.
void caseAdapterCpuAdd() {
    const std::string label = "adapter_cpu_add_semantics";
    auto A = makeTensor("A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto B = makeTensor("B", {4}, {0.5f, 1.5f, 2.5f, 3.5f});
    auto C = zeroTensor("C", {4});
    adapterAdd(A, B, C);
    if (!checkTensor(label, C, {1.5f, 3.5f, 5.5f, 7.5f}, 1e-6f)) return;
    std::cout << "PASS[" << label << "]\n";
}

// --- end-to-end parity tests -----------------------------------------------

std::unique_ptr<tsy::lir::Module> buildLir(const std::string& path,
                                            tsy::DiagnosticEngine& diag) {
    auto pr = tsy::parseFile(path);
    if (!pr.ok) { pr.diagnostics.print(std::cerr); return nullptr; }
    auto hmod = tsy::hir::lowerAstToHir(*pr.ast, pr.diagnostics);
    if (!hmod || pr.diagnostics.hasErrors()) {
        pr.diagnostics.print(std::cerr);
        return nullptr;
    }
    tsy::hir::verifyModule(*hmod, pr.diagnostics);
    if (pr.diagnostics.hasErrors()) {
        pr.diagnostics.print(std::cerr);
        return nullptr;
    }
    return tsy::lir::lowerHirToLir(*hmod, diag);
}

void caseParity(const std::string& dir, const std::string& tsy_name, float atol) {
    const std::string label = "adapter_cpu_vs_interp_" + tsy_name;
    tsy::DiagnosticEngine diag;
    auto lmod = buildLir(dir + "/" + tsy_name + ".tsy", diag);
    if (!lmod) { fail(label, "lowering failed"); return; }

    auto nativeRun = tsy::lir::runFirstTensorFunction(*lmod, diag);
    auto adapterRun = tsy::runtime::runWithCpuAdapter(*lmod, diag);
    if (!nativeRun.ok || !adapterRun.ok) {
        fail(label, "one of the backends refused to run");
        return;
    }
    if (nativeRun.buffers.size() != adapterRun.buffers.size()) {
        fail(label, "buffer count mismatch");
        return;
    }
    for (size_t b = 0; b < nativeRun.buffers.size(); ++b) {
        const auto& n = nativeRun.buffers[b];
        const auto& a = adapterRun.buffers[b];
        if (n.data.size() != a.data.size()) {
            fail(label, "shape mismatch on buffer " + n.name);
            return;
        }
        for (size_t i = 0; i < n.data.size(); ++i) {
            if (!approx(n.data[i], a.data[i], atol)) {
                std::ostringstream oss;
                oss << n.name << "[" << i << "] native=" << n.data[i]
                    << " adapter=" << a.data[i] << " (atol=" << atol << ")";
                fail(label, oss.str());
                return;
            }
        }
    }
    std::cout << "PASS[" << label << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_adapter_cpu_cases <examples_dir>\n";
        return 2;
    }
    const std::string dir = argv[1];

    caseAdapterCpuMatmulLayout();
    caseAdapterCpuSoftmaxDim();
    caseAdapterCpuRmsnormLastDim();
    caseAdapterCpuAdd();

    // End-to-end parity on every W4 example. Matmul is identical modulo
    // accumulation order; softmax and rmsnorm may diverge by a few ulps
    // because the two impls sum in different orders.
    caseParity(dir, "run_matmul_tiny", 1e-5f);
    caseParity(dir, "run_add_tiny", 1e-6f);
    caseParity(dir, "run_softmax_tiny", 1e-5f);
    caseParity(dir, "run_rmsnorm_tiny", 1e-4f);
    caseParity(dir, "run_matmul_odd", 1e-5f);
    // W7 MLP example: native interpreter and CPU adapter must agree on the
    // whole matmul→add→softmax chain. This doubles as the "interpreter /
    // codegen 对拍" check for PLAN.md §W7 because the codegen emits the
    // same adapter calls.
    caseParity(dir, "mlp", 1e-4f);

    if (g_failures) {
        std::cerr << "\n" << g_failures << " case(s) failed.\n";
        return 1;
    }
    std::cout << "\nall adapter_cpu cases passed.\n";
    return 0;
}
