#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "../frontend/diagnostics.h"
#include "ir.h"

namespace tsy::lir {

// One materialised tensor after a run: the raw fp32 values plus enough
// metadata for tests and CLI dumps to interpret them.
struct NamedTensor {
    std::string name;
    std::vector<int64_t> dims;
    std::vector<float> data;   // row-major.
    bool is_param = false;
};

struct RunResult {
    std::string function_name;
    std::vector<NamedTensor> buffers;
    bool ok = true;            // false if any op refused to run.
};

// Fill rule used by run-lir for parameter buffers. Exposed so tests can
// reproduce the inputs independently of the interpreter state.
float deterministicValue(int buf_idx, int64_t elem_idx);
void fillDeterministic(NamedTensor& t, int buf_idx);

// Run the first non-`main` function that has at least one tensor param. If
// the file has no such function, falls back to the first function in order.
RunResult runFirstTensorFunction(const Module& m, tsy::DiagnosticEngine& diag);

// Run a specific function by name (used by the test driver).
RunResult runNamedFunction(const Module& m, const std::string& name,
                           tsy::DiagnosticEngine& diag);

void printRunResult(std::ostream& os, const RunResult& r);

}  // namespace tsy::lir
