// W9: empty stub — linker needs this symbol to exist so Task 6's CMake
// change and the PassManager registration compile and link. Task 11
// replaces the body with the real shape-lookup.

#include "pass_manager.h"

namespace tsy::passes {

void runScheduleCuda(tsy::lir::Module& /*m*/,
                     tsy::DiagnosticEngine& /*diag*/) {
    // intentionally empty — Task 11 replaces this.
}

}  // namespace tsy::passes
