#include "module_utils.h"

namespace tsy::lir {

const Function* pickFirstTensorFunction(const Module& m) {
    for (const auto& f : m.funcs) {
        if (f->name == "main") continue;
        if (!f->params.empty()) return f.get();
    }
    return m.funcs.empty() ? nullptr : m.funcs.front().get();
}

}  // namespace tsy::lir
