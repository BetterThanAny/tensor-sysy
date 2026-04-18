#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "location.h"

namespace tsy {

enum class DiagLevel { Note, Warning, Error, Fatal };

struct Diagnostic {
    DiagLevel level = DiagLevel::Error;
    SourceLocation loc;
    std::string message;
};

class DiagnosticEngine {
   public:
    DiagnosticEngine() = default;

    void report(DiagLevel level, SourceLocation loc, std::string msg);
    void error(SourceLocation loc, std::string msg) { report(DiagLevel::Error, std::move(loc), std::move(msg)); }
    void warning(SourceLocation loc, std::string msg) { report(DiagLevel::Warning, std::move(loc), std::move(msg)); }

    const std::vector<Diagnostic>& diagnostics() const { return diags_; }
    bool hasErrors() const { return error_count_ > 0; }
    int errorCount() const { return error_count_; }

    void print(std::ostream& os) const;

   private:
    std::vector<Diagnostic> diags_;
    int error_count_ = 0;
};

const char* diagLevelLabel(DiagLevel level);

}  // namespace tsy
