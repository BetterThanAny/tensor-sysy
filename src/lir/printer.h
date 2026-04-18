#pragma once

#include <ostream>

#include "ir.h"

namespace tsy::lir {

void printModule(std::ostream& os, const Module& m);
void printFunction(std::ostream& os, const Function& f, int indent = 2);

}  // namespace tsy::lir
