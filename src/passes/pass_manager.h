#pragma once

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include "../frontend/diagnostics.h"
#include "../hir/ops.h"

namespace tsy::passes {

// A pass transforms an HIR module in place. Diagnostics go through the
// shared engine so `tsc emit-hir` and test drivers see the same messages.
using PassFn = std::function<void(tsy::hir::Module&, tsy::DiagnosticEngine&)>;

class PassManager {
   public:
    // Register a pass under a user-visible name. Names are matched by the
    // `--disable-pass=<name>` CLI flag and by the test driver.
    void add(std::string name, PassFn fn);

    // Skip a pass by name for this run (or until reenable). Unknown names
    // are silently tolerated so disabling a pass that isn't in the current
    // pipeline is a no-op.
    void disable(const std::string& name);
    void enable(const std::string& name);
    bool isDisabled(const std::string& name) const;

    // Return the ordered pass names (useful for logs / CLI dumps).
    std::vector<std::string> names() const;

    void run(tsy::hir::Module& m, tsy::DiagnosticEngine& diag) const;

   private:
    struct Entry { std::string name; PassFn fn; };
    std::vector<Entry> passes_;
    std::unordered_set<std::string> disabled_;
};

// Individual passes exposed for tests / custom pipelines.
void runVerifier(tsy::hir::Module& m, tsy::DiagnosticEngine& diag);
void runConstFold(tsy::hir::Module& m, tsy::DiagnosticEngine& diag);
void runDCE(tsy::hir::Module& m, tsy::DiagnosticEngine& diag);

// --- standard pipelines ---------------------------------------------------
//
// O0: verifier only (structural safety net, no transforms).
// O1: verifier → const-fold → dce → verifier (re-verify after changes).
PassManager buildPipelineO0();
PassManager buildPipelineO1();

}  // namespace tsy::passes
