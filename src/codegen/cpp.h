#pragma once

#include <ostream>
#include <string>

#include "../lir/ir.h"

namespace tsy::codegen {

// Emit a self-contained C++ program whose behaviour reproduces running the
// first tensor function in `m` through the CPU adapter (W6 backend).
//
// The generated file only depends on two headers exposed by the
// `tsy_runtime_cpu` library: `<adapter_cpu.h>` for the primitive dispatch
// and `<interpreter.h>` for `NamedTensor`, `fillDeterministic`, and
// `printRunResult`. This lets W7 codegen stay a thin translation layer
// instead of re-inventing the entire runtime; later weeks that care about
// a standalone artifact (e.g. for a PyTorch-less inspection build) can
// inline those types, but the plan only asks for a CPU end-to-end loop.
//
// Returns true on success. The function picks the first non-`main`
// function with tensor parameters; if none exists, emits an empty main().
bool emitCppModule(std::ostream& os, const tsy::lir::Module& m,
                   const std::string& source_path);

}  // namespace tsy::codegen
