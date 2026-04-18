#include "pass_manager.h"

namespace tsy::passes {

void PassManager::add(std::string name, PassFn fn) {
    passes_.push_back({std::move(name), std::move(fn)});
}

void PassManager::disable(const std::string& name) { disabled_.insert(name); }
void PassManager::enable(const std::string& name) { disabled_.erase(name); }
bool PassManager::isDisabled(const std::string& name) const {
    return disabled_.count(name) != 0;
}

std::vector<std::string> PassManager::names() const {
    std::vector<std::string> out;
    out.reserve(passes_.size());
    for (const auto& e : passes_) out.push_back(e.name);
    return out;
}

void PassManager::run(tsy::hir::Module& m, tsy::DiagnosticEngine& diag) const {
    for (const auto& e : passes_) {
        if (disabled_.count(e.name)) continue;
        e.fn(m, diag);
        if (diag.hasErrors()) break;  // stop pipeline on any hard error.
    }
}

PassManager buildPipelineO0() {
    PassManager pm;
    pm.add("verify", runVerifier);
    return pm;
}

PassManager buildPipelineO1() {
    PassManager pm;
    pm.add("verify", runVerifier);
    pm.add("const-fold", runConstFold);
    pm.add("dce", runDCE);
    pm.add("verify-post", runVerifier);  // re-verify after transforms.
    return pm;
}

}  // namespace tsy::passes
