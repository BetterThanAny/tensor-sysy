#pragma once

#include "ir.h"

namespace tsy::lir {

// Pick the first "interesting" function to execute / codegen.
// Skips `main`, prefers functions with non-empty tensor params, falls back to
// module.funcs.front() so run-lir always has something to do. Previously
// duplicated verbatim across interpreter, adapter_cpu, adapter_cuda, and
// both codegen modules; this is now the single authoritative copy.
const Function* pickFirstTensorFunction(const Module& m);

}  // namespace tsy::lir
