#pragma once

#include <memory>

#include "../frontend/ast.h"
#include "../frontend/diagnostics.h"
#include "ops.h"

namespace tsy::hir {

// Lower a parsed AST's CompUnit into a tensor HIR Module.
//
// W2 scope:
//   - Function signatures (tensor + scalar params, int/void return).
//   - Tensor var decls whose initializer is a single @builtin(...) call on
//     lvalue arguments become typed ops in the function body.
//   - `return;` / `return <expr>;` emit a Return op.
//   - Anything else (nested builtins, arithmetic on tensors, if/while...) is
//     emitted as an OpKind::Unknown carrying the reason — the HIR dump stays
//     readable and later weeks can grow the coverage without API churn.
//
// Dims on tensor types are symbolic (original source text). W3 adds a const
// evaluator that resolves them.
std::unique_ptr<Module> lowerAstToHir(const tsy::BaseAST& compUnit,
                                      tsy::DiagnosticEngine& diag);

}  // namespace tsy::hir
