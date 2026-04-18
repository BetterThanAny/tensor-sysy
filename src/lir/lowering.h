#pragma once

#include <memory>

#include "../frontend/diagnostics.h"
#include "../hir/ops.h"
#include "ir.h"

namespace tsy::lir {

// Lower an already-verified HIR module to the primitive-call LIR. W4 only
// emits buffers/calls for the four recognised builtins and `return`; other
// ops are elided with a diagnostic so the interpreter can refuse them.
std::unique_ptr<Module> lowerHirToLir(const tsy::hir::Module& hir,
                                      tsy::DiagnosticEngine& diag);

}  // namespace tsy::lir
