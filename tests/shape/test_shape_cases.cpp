// W3 shape / verifier cases.
//
// Drives the full parse → lower → verify pipeline on fixture files and
// asserts on the resulting diagnostics for bad inputs, or on the printed
// HIR for good ones (checking that dims actually got resolved).

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lowering.h"
#include "parser_driver.h"
#include "printer.h"
#include "verifier.h"

namespace {

struct Case {
    std::string label;
    std::string file;
    bool expect_ok;
    std::string diag_substr;   // expected substring in stderr-like diag text
    int expect_line;           // 0 = skip
    std::string hir_substr;    // for good cases: substring of printed HIR
};

int g_failures = 0;

void fail(const std::string& label, const std::string& why) {
    std::cerr << "FAIL[" << label << "]: " << why << "\n";
    ++g_failures;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return !needle.empty() && haystack.find(needle) != std::string::npos;
}

std::string dumpDiagnostics(const tsy::DiagnosticEngine& d) {
    std::ostringstream oss;
    d.print(oss);
    return oss.str();
}

void run(const Case& c) {
    auto r = tsy::parseFile(c.file);
    if (!r.ok) {
        if (c.expect_ok) {
            fail(c.label, "parse failed unexpectedly:\n" + dumpDiagnostics(r.diagnostics));
            return;
        }
        // Parse-level failure is still a failure — check substring if set.
        if (!c.diag_substr.empty() && !contains(dumpDiagnostics(r.diagnostics), c.diag_substr)) {
            fail(c.label, "expected '" + c.diag_substr + "' in parse diag:\n" +
                              dumpDiagnostics(r.diagnostics));
            return;
        }
        std::cout << "PASS[" << c.label << "]\n";
        return;
    }

    auto mod = tsy::hir::lowerAstToHir(*r.ast, r.diagnostics);
    if (mod && !r.diagnostics.hasErrors()) {
        tsy::hir::verifyModule(*mod, r.diagnostics);
    }
    const bool ok = (mod != nullptr) && !r.diagnostics.hasErrors();
    const auto diagText = dumpDiagnostics(r.diagnostics);

    if (c.expect_ok && !ok) {
        fail(c.label, "expected success, got diagnostics:\n" + diagText);
        return;
    }
    if (!c.expect_ok && ok) {
        fail(c.label, "expected failure, got clean pass");
        return;
    }
    if (!c.diag_substr.empty() && !contains(diagText, c.diag_substr)) {
        fail(c.label, "expected diag substring '" + c.diag_substr + "' not found in:\n" + diagText);
        return;
    }
    if (c.expect_line > 0) {
        bool matched = false;
        for (const auto& d : r.diagnostics.diagnostics()) {
            if (d.loc.line == c.expect_line) { matched = true; break; }
        }
        if (!matched) {
            fail(c.label, "no diagnostic on line " + std::to_string(c.expect_line) +
                              "; got:\n" + diagText);
            return;
        }
    }
    if (c.expect_ok && !c.hir_substr.empty()) {
        std::ostringstream hirSS;
        tsy::hir::printModule(hirSS, *mod);
        if (!contains(hirSS.str(), c.hir_substr)) {
            fail(c.label, "HIR missing substring '" + c.hir_substr + "':\n" + hirSS.str());
            return;
        }
    }
    std::cout << "PASS[" << c.label << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_shape_cases <examples_dir>\n";
        return 2;
    }
    const std::string base = argv[1];

    std::vector<Case> cases = {
        // --- good: dims must be resolved through const eval -----------------
        {"shape_matmul_resolved", base + "/matmul.tsy", true, "", 0,
         "tensor<f32>[M=8, N=4]"},
        {"shape_matmul_inner_matches", base + "/matmul.tsy", true, "", 0,
         "tensor<f32>[K=16, N=4]"},
        {"shape_all_ops_resolved", base + "/tensor_all_ops.tsy", true, "", 0,
         "tensor<f32>[M=4, N=4]"},

        // --- bad: shape / arity / unknown builtin / unresolved --------------
        {"shape_bad_matmul_mismatch", base + "/bad_matmul_mismatch.tsy", false,
         "inner dim mismatch", 8, ""},
        {"shape_bad_add_shape", base + "/bad_add_shape.tsy", false,
         "identical shapes", 7, ""},
        {"shape_bad_softmax_arity", base + "/bad_softmax_arity.tsy", false,
         "softmax expects 1 operand", 7, ""},
        {"shape_bad_rmsnorm_result", base + "/bad_rmsnorm_result_shape.tsy",
         false, "rmsnorm result shape", 8, ""},
        {"shape_bad_unknown_builtin", base + "/bad_unknown_builtin.tsy", false,
         "unknown builtin '@gelu'", 6, ""},
        {"shape_bad_unresolved_dim", base + "/bad_unresolved_dim.tsy", false,
         "unknown constant 'M'", 0, ""},
    };

    for (const auto& c : cases) run(c);

    if (g_failures) {
        std::cerr << "\n" << g_failures << " case(s) failed.\n";
        return 1;
    }
    std::cout << "\nall shape cases passed.\n";
    return 0;
}
