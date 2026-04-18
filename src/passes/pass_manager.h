#pragma once

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include "../frontend/diagnostics.h"
#include "../hir/ops.h"
#include "../lir/ir.h"

namespace tsy::passes {

// A pass transforms an HIR module in place. Diagnostics go through the
// shared engine so `tsc emit-hir` and test drivers see the same messages.
using PassFn = std::function<void(tsy::hir::Module&, tsy::DiagnosticEngine&)>;

using LirPassFn =
    std::function<void(tsy::lir::Module&, tsy::DiagnosticEngine&)>;

class PassManager {
   public:
    void add(std::string name, PassFn fn);
    void addLir(std::string name, LirPassFn fn);

    void disable(const std::string& name);
    void enable(const std::string& name);
    bool isDisabled(const std::string& name) const;

    std::vector<std::string> names() const;     // HIR + LIR combined
    std::vector<std::string> lirNames() const;  // LIR-only subset

    void run(tsy::hir::Module& m, tsy::DiagnosticEngine& diag) const;
    void runLir(tsy::lir::Module& m, tsy::DiagnosticEngine& diag) const;

   private:
    struct HirEntry { std::string name; PassFn fn; };
    struct LirEntry { std::string name; LirPassFn fn; };
    std::vector<HirEntry> passes_;
    std::vector<LirEntry> lir_passes_;
    std::unordered_set<std::string> disabled_;
};

// Individual passes exposed for tests / custom pipelines.
void runVerifier(tsy::hir::Module& m, tsy::DiagnosticEngine& diag);
void runConstFold(tsy::hir::Module& m, tsy::DiagnosticEngine& diag);
void runDCE(tsy::hir::Module& m, tsy::DiagnosticEngine& diag);

// LIR passes (W9).
void runLayoutLowering(tsy::lir::Module& m, tsy::DiagnosticEngine& diag);
void runScheduleCuda(tsy::lir::Module& m, tsy::DiagnosticEngine& diag);

// --- standard pipelines ---------------------------------------------------
//
// O0: verifier only (structural safety net, no transforms).
// O1: verifier → const-fold → dce → verifier (re-verify after changes).
PassManager buildPipelineO0();
PassManager buildPipelineO1();

}  // namespace tsy::passes
