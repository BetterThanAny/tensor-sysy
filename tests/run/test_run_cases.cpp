// W4 interpreter cases.
//
// Runs parse → HIR lower → verify → LIR lower → interpreter in-process on
// fixture files, then compares fp32 outputs against hand-computed
// references. Acts as the "naive interpreter vs expected values" smoke —
// the numpy reference script is additive and optional for CI.

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/hir/lowering.h"
#include "../../src/hir/verifier.h"
#include "../../src/lir/interpreter.h"
#include "../../src/lir/lowering.h"
#include "../../src/frontend/parser_driver.h"

namespace {

int g_failures = 0;

void fail(const std::string& label, const std::string& why) {
    std::cerr << "FAIL[" << label << "]: " << why << "\n";
    ++g_failures;
}

bool approxEq(float a, float b, float atol = 1e-5f) {
    return std::fabs(a - b) <= atol;
}

const tsy::lir::NamedTensor* findBuffer(const tsy::lir::RunResult& r,
                                        const std::string& name) {
    for (const auto& t : r.buffers) if (t.name == name) return &t;
    return nullptr;
}

tsy::lir::RunResult runFile(const std::string& path) {
    tsy::lir::RunResult empty;
    auto pr = tsy::parseFile(path);
    if (!pr.ok) {
        std::ostringstream oss;
        pr.diagnostics.print(oss);
        std::cerr << "parse failed for " << path << ":\n" << oss.str();
        return empty;
    }
    auto hmod = tsy::hir::lowerAstToHir(*pr.ast, pr.diagnostics);
    if (!hmod || pr.diagnostics.hasErrors()) { return empty; }
    tsy::hir::verifyModule(*hmod, pr.diagnostics);
    if (pr.diagnostics.hasErrors()) { return empty; }
    auto lmod = tsy::lir::lowerHirToLir(*hmod, pr.diagnostics);
    if (!lmod || pr.diagnostics.hasErrors()) { return empty; }
    return tsy::lir::runFirstTensorFunction(*lmod, pr.diagnostics);
}

// Check that every element of `actual` is approximately equal to
// the corresponding element of `expected` within `atol`.
bool matchTensor(const std::string& label, const tsy::lir::NamedTensor& t,
                 const std::vector<float>& expected, float atol = 1e-5f) {
    if (t.data.size() != expected.size()) {
        fail(label, "size mismatch: got " + std::to_string(t.data.size()) +
                        ", expected " + std::to_string(expected.size()));
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!approxEq(t.data[i], expected[i], atol)) {
            std::ostringstream oss;
            oss << "element " << i << " differs: got " << t.data[i]
                << ", expected " << expected[i] << " (atol=" << atol << ")";
            fail(label, oss.str());
            return false;
        }
    }
    return true;
}

// --- cases -----------------------------------------------------------------

void caseMatmulTiny(const std::string& dir) {
    const std::string label = "run_matmul_tiny";
    auto r = runFile(dir + "/run_matmul_tiny.tsy");
    if (!r.ok) { fail(label, "runFile not ok"); return; }
    const auto* A = findBuffer(r, "A");
    const auto* B = findBuffer(r, "B");
    const auto* C = findBuffer(r, "C");
    if (!A || !B || !C) { fail(label, "missing A/B/C buffer"); return; }
    if (!matchTensor(label + ":A", *A, {0.0f, 0.1f, 0.2f, 0.3f})) return;
    if (!matchTensor(label + ":B", *B, {0.5f, 0.6f, 0.7f, 0.8f})) return;
    if (!matchTensor(label + ":C", *C, {0.07f, 0.08f, 0.31f, 0.36f})) return;
    std::cout << "PASS[" << label << "]\n";
}

void caseAddTiny(const std::string& dir) {
    const std::string label = "run_add_tiny";
    auto r = runFile(dir + "/run_add_tiny.tsy");
    if (!r.ok) { fail(label, "runFile not ok"); return; }
    const auto* A = findBuffer(r, "A");
    const auto* B = findBuffer(r, "B");
    const auto* C = findBuffer(r, "C");
    if (!A || !B || !C) { fail(label, "missing A/B/C buffer"); return; }
    // A = [0.0, 0.1, 0.2, 0.3]; B = [0.5, 0.6, 0.7, 0.8]; C = A + B.
    std::vector<float> expected = {0.5f, 0.7f, 0.9f, 1.1f};
    if (!matchTensor(label + ":C", *C, expected)) return;
    std::cout << "PASS[" << label << "]\n";
}

// Softmax along last dim: for a 1x4 input x, output[i] = exp(x[i]-max) / sum.
// With inputs [0.0, 0.1, 0.2, 0.3] and max=0.3:
//   e0 = exp(-0.3), e1 = exp(-0.2), e2 = exp(-0.1), e3 = exp(0) = 1.
// Sum ≈ 0.7408 + 0.8187 + 0.9048 + 1.0000 = 3.4644.
// Normalised ≈ [0.2138, 0.2363, 0.2612, 0.2887].
void caseSoftmaxTiny(const std::string& dir) {
    const std::string label = "run_softmax_tiny";
    auto r = runFile(dir + "/run_softmax_tiny.tsy");
    if (!r.ok) { fail(label, "runFile not ok"); return; }
    const auto* Y = findBuffer(r, "Y");
    if (!Y) { fail(label, "missing Y"); return; }
    std::vector<float> expected = {0.2138f, 0.2363f, 0.2612f, 0.2887f};
    if (!matchTensor(label + ":Y", *Y, expected, 1e-4f)) return;
    // Sums along the last dim must be 1.
    float sum = 0.0f;
    for (float v : Y->data) sum += v;
    if (!approxEq(sum, 1.0f, 1e-4f)) {
        fail(label, "softmax sum not 1, got " + std::to_string(sum));
        return;
    }
    std::cout << "PASS[" << label << "]\n";
}

// RMSNorm along last dim with eps=1e-6. For a 1x4 input [0.0, 0.1, 0.2, 0.3]:
//   mean-square = (0 + 0.01 + 0.04 + 0.09)/4 = 0.035
//   rms = sqrt(0.035 + 1e-6) ≈ 0.18708
//   out = x / rms ≈ [0.0, 0.5345, 1.0690, 1.6036]
void caseRmsnormTiny(const std::string& dir) {
    const std::string label = "run_rmsnorm_tiny";
    auto r = runFile(dir + "/run_rmsnorm_tiny.tsy");
    if (!r.ok) { fail(label, "runFile not ok"); return; }
    const auto* Y = findBuffer(r, "Y");
    if (!Y) { fail(label, "missing Y"); return; }
    std::vector<float> expected = {0.0f, 0.5345f, 1.0690f, 1.6036f};
    if (!matchTensor(label + ":Y", *Y, expected, 1e-3f)) return;
    std::cout << "PASS[" << label << "]\n";
}

// Odd-shaped matmul to confirm stride math: 1x3 @ 3x1.
// A = [0.0, 0.1, 0.2] at buf_idx 0; B = [0.5, 0.6, 0.7] at buf_idx 1.
// C = A @ B = 0*0.5 + 0.1*0.6 + 0.2*0.7 = 0.06 + 0.14 = 0.20.
void caseMatmulOddShape(const std::string& dir) {
    const std::string label = "run_matmul_odd";
    auto r = runFile(dir + "/run_matmul_odd.tsy");
    if (!r.ok) { fail(label, "runFile not ok"); return; }
    const auto* C = findBuffer(r, "C");
    if (!C) { fail(label, "missing C"); return; }
    if (!matchTensor(label + ":C", *C, {0.20f}, 1e-4f)) return;
    std::cout << "PASS[" << label << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_run_cases <examples_dir>\n";
        return 2;
    }
    const std::string dir = argv[1];
    caseMatmulTiny(dir);
    caseAddTiny(dir);
    caseSoftmaxTiny(dir);
    caseRmsnormTiny(dir);
    caseMatmulOddShape(dir);
    if (g_failures) {
        std::cerr << "\n" << g_failures << " case(s) failed.\n";
        return 1;
    }
    std::cout << "\nall run cases passed.\n";
    return 0;
}
