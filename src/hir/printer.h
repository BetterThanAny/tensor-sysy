#pragma once

#include <ostream>

#include "ops.h"

namespace tsy::hir {

void printModule(std::ostream& os, const Module& m);
void printFunction(std::ostream& os, const Function& f, int indent = 2);
void printOp(std::ostream& os, const Op& op, int indent = 4);

}  // namespace tsy::hir
