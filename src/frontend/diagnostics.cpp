#include "diagnostics.h"

namespace tsy {

const char* diagLevelLabel(DiagLevel level) {
    switch (level) {
        case DiagLevel::Note: return "note";
        case DiagLevel::Warning: return "warning";
        case DiagLevel::Error: return "error";
        case DiagLevel::Fatal: return "fatal";
    }
    return "error";
}

void DiagnosticEngine::report(DiagLevel level, SourceLocation loc, std::string msg) {
    diags_.push_back({level, std::move(loc), std::move(msg)});
    if (level == DiagLevel::Error || level == DiagLevel::Fatal) ++error_count_;
}

void DiagnosticEngine::print(std::ostream& os) const {
    for (const auto& d : diags_) {
        os << d.loc.format() << ": " << diagLevelLabel(d.level) << ": " << d.message << "\n";
    }
}

}  // namespace tsy
