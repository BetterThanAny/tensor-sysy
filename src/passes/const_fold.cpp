#include "pass_manager.h"

namespace tsy::passes {

// Constant folding on tensor HIR.
//
// W5 intentionally leaves this pass as a structural no-op. The tensor
// HIR at this point only carries ops that consume tensor *values* that
// entered the module as function parameters; we don't yet materialise
// constant tensors (`hir::OpKind::Const` is reserved but unemitted) or
// host-side scalars that could be folded into op attributes. Classic
// folding opportunities therefore don't apply.
//
// Scalar const evaluation (the `M`, `K`, `N` style identifiers used
// inside tensor shapes) was already handled at lowering time by
// src/hir/const_eval.cpp, so there is nothing to re-fold here either.
//
// The pass still exists so the PassManager pipeline in W5 has a named
// slot to register future rewrites against. Once W7's C++ codegen grows
// tensor-literal support, this file is where `add(x, zero) -> x`-style
// identities will live.
void runConstFold(tsy::hir::Module& /*m*/, tsy::DiagnosticEngine& /*diag*/) {
    // deliberately empty — see banner above for the "why".
}

}  // namespace tsy::passes
