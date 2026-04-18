#pragma once

#include <ostream>
#include <string>

#include "../lir/ir.h"

namespace tsy::codegen {

// Write a self-contained .cu source that runs the LIR module through
// adapter_cuda at runtime. The generated binary is expected to link
// against tsy_runtime_cuda. The output is pure host C++ with .cu
// extension so nvcc handles the one-step compile + link.
bool emitCudaModule(std::ostream& os, const tsy::lir::Module& m,
                    const std::string& source_path);

}  // namespace tsy::codegen
