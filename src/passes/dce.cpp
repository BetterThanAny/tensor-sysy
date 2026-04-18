#include <algorithm>
#include <unordered_set>

#include "pass_manager.h"

namespace tsy::passes {

namespace {

// Collect every HIR Value that is currently consumed as an operand. These
// are the values that MUST stay around. Everything else whose only user is
// itself is dead.
std::unordered_set<const tsy::hir::Value*> collectLive(
    const tsy::hir::Function& f) {
    std::unordered_set<const tsy::hir::Value*> live;
    for (const auto& opPtr : f.ops) {
        for (const auto& v : opPtr->operands) live.insert(v.get());
    }
    return live;
}

bool dropDeadOps(tsy::hir::Function& f) {
    const auto live = collectLive(f);
    auto& ops = f.ops;
    const auto before = ops.size();
    ops.erase(std::remove_if(ops.begin(), ops.end(),
        [&](const std::unique_ptr<tsy::hir::Op>& opPtr) {
            const auto& op = *opPtr;
            // Terminators and failure markers stay no matter what.
            if (op.kind == tsy::hir::OpKind::Return) return false;
            if (op.kind == tsy::hir::OpKind::Unknown) return false;
            // Ops that don't produce results also stay — they must have
            // side effects we don't yet model.
            if (op.results.empty()) return false;
            for (const auto& r : op.results) {
                if (live.count(r.get())) return false;
            }
            return true;
        }), ops.end());
    return ops.size() != before;
}

}  // namespace

// Dead-code elimination for tensor HIR.
//
// Algorithm:
//   1. Build a set of values currently used as operands (the "live set").
//   2. Remove any op whose results are all absent from the live set,
//      except Return (terminator) and Unknown (preserves diagnostics for
//      unhandled AST shapes).
//   3. Iterate to a fixed point — removing one op can render its producer
//      inputs unused and let the next pass delete them too.
//
// DCE is intentionally conservative: we never drop ops that have no
// results (they might be side-effectful calls in later weeks) and we keep
// the full original operand->producer chain intact, since HIR uses
// shared_ptr<Value> so the producer Op carries no implicit owner.
void runDCE(tsy::hir::Module& m, tsy::DiagnosticEngine& /*diag*/) {
    for (auto& f : m.funcs) {
        while (dropDeadOps(*f)) { /* keep iterating until fixed point */ }
    }
}

}  // namespace tsy::passes
