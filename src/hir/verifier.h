#pragma once

#include "../frontend/diagnostics.h"
#include "ops.h"

namespace tsy::hir {

// Run every structural check we can do on the HIR without lowering further.
// Issues are reported through `diag`; the module is not mutated.
//
// W3 checks:
//   - MatMul: 2 operands, both tensors rank ≥ 2, inner dims match, result
//     shape is [lhs[:-1]..., rhs[-1]].
//   - Add: 2 operands, same rank and shape, same dtype.
//   - Softmax / RMSNorm: 1 operand, result shape equals operand shape.
//   - Unknown ops: report a best-effort "unrecognised builtin" diagnostic
//     that carries the AST name preserved by the lowerer.
//   - Shape integrity: tensor operands must have fully resolved dims.
void verifyModule(const Module& m, tsy::DiagnosticEngine& diag);

}  // namespace tsy::hir
