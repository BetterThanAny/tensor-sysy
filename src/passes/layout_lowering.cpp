// W9: no-op placeholder. W10 fills in real layout transformations.
//
// Where this pass is going (W10 transformer block):
//   - Recognise non-canonical matmul operand layouts and emit transposes
//     that make them canonical before ScheduleCudaPass picks a kernel.
//   - Expand View/Permute ops (HIR's currently-reserved enums) into
//     adjacent buffer reshape + copy statements.
//
// For W9 this pass is registered in PassManager O1 pipeline so the
// structural hookup is done once. W10 can drop in the body here without
// touching pass_manager / CLI / tsc.

#include "pass_manager.h"

namespace tsy::passes {

void runLayoutLowering(tsy::lir::Module& /*m*/,
                       tsy::DiagnosticEngine& /*diag*/) {
    // intentionally empty
}

}  // namespace tsy::passes
