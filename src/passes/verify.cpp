#include "../hir/verifier.h"
#include "pass_manager.h"

namespace tsy::passes {

// Wrapper around the in-tree HIR verifier so the PassManager pipeline can
// treat it like any other pass (ordered, nameable, disable-able).
void runVerifier(tsy::hir::Module& m, tsy::DiagnosticEngine& diag) {
    tsy::hir::verifyModule(m, diag);
}

}  // namespace tsy::passes
