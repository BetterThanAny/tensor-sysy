// W0 parse / diagnostics smoke suite.
//
// Runs a hand-rolled list of cases against tsy::parseFile() and exits nonzero
// on any mismatch. Intentionally avoids a test framework dependency for W0 —
// when we grow more assertion complexity, swap this for gtest.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "parser_driver.h"

namespace {

struct Case {
    std::string label;
    std::string file;
    bool expect_ok;
    std::string diag_substr;  // empty = don't check
    int expect_error_line;    // 0 = don't check
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
    const auto diag_text = dumpDiagnostics(r.diagnostics);

    if (c.expect_ok && !r.ok) {
        fail(c.label, "expected parse ok, got failure:\n" + diag_text);
        return;
    }
    if (!c.expect_ok && r.ok) {
        fail(c.label, "expected parse failure, got success");
        return;
    }
    if (!c.diag_substr.empty() && !contains(diag_text, c.diag_substr)) {
        fail(c.label, "expected diagnostic substring '" + c.diag_substr +
                          "' not found in:\n" + diag_text);
        return;
    }
    if (c.expect_error_line > 0) {
        bool found = false;
        for (const auto& d : r.diagnostics.diagnostics()) {
            if (d.loc.line == c.expect_error_line) {
                found = true;
                break;
            }
        }
        if (!found) {
            fail(c.label, "expected diagnostic on line " +
                              std::to_string(c.expect_error_line) + "; got:\n" +
                              diag_text);
            return;
        }
    }
    std::cout << "PASS[" << c.label << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_parse_cases <examples_dir>\n";
        return 2;
    }
    const std::string base = argv[1];

    std::vector<Case> cases = {
        // W0 — original SysY must keep parsing after W1's tensor additions.
        {"smoke_parses_ok", base + "/smoke.tsy", true, "", 0},
        {"bad_syntax_fails", base + "/bad_syntax.tsy", false,
         "parse error", 0},
        {"bad_syntax_has_line", base + "/bad_syntax.tsy", false,
         "bad_syntax.tsy:", 0},

        // W1 — tensor syntax and builtin ops must parse successfully.
        {"matmul_parses_ok", base + "/matmul.tsy", true, "", 0},
        {"tensor_all_ops_ok", base + "/tensor_all_ops.tsy", true, "", 0},

        // W1 — tensor / builtin diagnostics must report the offending line.
        {"bad_tensor_missing_dtype_fails",
         base + "/bad_tensor_missing_dtype.tsy", false, "parse error", 0},
        {"bad_tensor_missing_dtype_has_file",
         base + "/bad_tensor_missing_dtype.tsy", false,
         "bad_tensor_missing_dtype.tsy:", 0},

        {"bad_tensor_missing_shape_fails",
         base + "/bad_tensor_missing_shape.tsy", false, "parse error", 0},
        {"bad_tensor_missing_shape_has_file",
         base + "/bad_tensor_missing_shape.tsy", false,
         "bad_tensor_missing_shape.tsy:", 0},

        {"bad_builtin_missing_parens_fails",
         base + "/bad_builtin_missing_parens.tsy", false, "parse error", 0},
        {"bad_builtin_missing_parens_has_file",
         base + "/bad_builtin_missing_parens.tsy", false,
         "bad_builtin_missing_parens.tsy:", 0},
    };

    for (const auto& c : cases) run(c);

    if (g_failures) {
        std::cerr << "\n" << g_failures << " case(s) failed.\n";
        return 1;
    }
    std::cout << "\nall parse cases passed.\n";
    return 0;
}
