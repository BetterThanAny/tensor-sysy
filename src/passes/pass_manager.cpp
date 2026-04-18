#include "pass_manager.h"

namespace tsy::passes {

void PassManager::add(std::string name, PassFn fn) {
    passes_.push_back({std::move(name), std::move(fn)});
}

void PassManager::addLir(std::string name, LirPassFn fn) {
    lir_passes_.push_back({std::move(name), std::move(fn)});
}

void PassManager::disable(const std::string& name) { disabled_.insert(name); }
void PassManager::enable(const std::string& name) { disabled_.erase(name); }
bool PassManager::isDisabled(const std::string& name) const {
    return disabled_.count(name) != 0;
}

std::vector<std::string> PassManager::names() const {
    std::vector<std::string> out;
    out.reserve(passes_.size() + lir_passes_.size());
    for (const auto& e : passes_) out.push_back(e.name);
    for (const auto& e : lir_passes_) out.push_back(e.name);
    return out;
}

std::vector<std::string> PassManager::lirNames() const {
    std::vector<std::string> out;
    out.reserve(lir_passes_.size());
    for (const auto& e : lir_passes_) out.push_back(e.name);
    return out;
}

void PassManager::run(tsy::hir::Module& m, tsy::DiagnosticEngine& diag) const {
    for (const auto& e : passes_) {
        if (disabled_.count(e.name)) continue;
        e.fn(m, diag);
        if (diag.hasErrors()) break;
    }
}

void PassManager::runLir(tsy::lir::Module& m,
                         tsy::DiagnosticEngine& diag) const {
    for (const auto& e : lir_passes_) {
        if (disabled_.count(e.name)) continue;
        e.fn(m, diag);
        if (diag.hasErrors()) break;
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
    pm.add("verify-post", runVerifier);
    // LIR passes run AFTER hir-to-lir lowering (callers handle that order).
    pm.addLir("layout-lowering", runLayoutLowering);
    pm.addLir("schedule-cuda", runScheduleCuda);
    return pm;
}

}  // namespace tsy::passes
