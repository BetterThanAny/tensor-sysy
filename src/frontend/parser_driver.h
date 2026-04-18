#pragma once

#include <optional>
#include <string>

#include "ast.h"
#include "diagnostics.h"

namespace tsy {

struct ParseResult {
    BaseASTPtr ast;
    DiagnosticEngine diagnostics;
    bool ok = false;
};

ParseResult parseFile(const std::string& path);

}  // namespace tsy
