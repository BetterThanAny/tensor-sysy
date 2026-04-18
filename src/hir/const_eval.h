#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "../frontend/ast.h"
#include "../frontend/diagnostics.h"

namespace tsy::hir {

// Mapping from `const int` identifier to its resolved integer value.
// Scope is intentionally flat for W3: only top-level `const int M = N;`
// decls are visible to tensor-dim const-eval. Locally-scoped constants or
// const-of-const chains will grow this as later weeks need it.
using ConstScope = std::unordered_map<std::string, int64_t>;

// Evaluate a constant expression (ConstExp / Exp / any subtree from the
// SysY grammar) to an integer. Returns std::nullopt and reports via diag
// when an identifier is unbound, when an unsupported form is encountered,
// or when an arithmetic error would occur (e.g. division by zero).
std::optional<int64_t> evalConstExp(const tsy::BaseAST& node,
                                    const ConstScope& scope,
                                    tsy::DiagnosticEngine& diag);

// Walk a parsed CompUnit and build a scope from every top-level
// `const int <name> = <expr>;` decl. Earlier entries are visible to later
// ones, matching the source-order expectation.
ConstScope collectGlobalConstScope(const tsy::BaseAST& compUnit,
                                   tsy::DiagnosticEngine& diag);

}  // namespace tsy::hir
