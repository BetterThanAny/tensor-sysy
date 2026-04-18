#include "parser_driver.h"

#include <cstdio>

#include "ast.h"
#include "diagnostics.h"
#include "sysy.tab.hpp"

extern FILE* yyin;
extern int yylineno;
void yyrestart(FILE*);  // flex-provided; flushes the scanner buffer.

// Declared by bison in sysy.tab.hpp; signature matches %parse-param order.
int yyparse(tsy::BaseASTPtr& ast, tsy::DiagnosticEngine& diag,
            const std::string& filename);

namespace tsy {

ParseResult parseFile(const std::string& path) {
    ParseResult result;
    FILE* fp = std::fopen(path.c_str(), "r");
    if (!fp) {
        result.diagnostics.error({path, 0, 0},
                                 std::string("cannot open file: ") + path);
        return result;
    }

    yyin = fp;
    yylineno = 1;
    yyrestart(fp);  // discard any buffered state from a prior parse.

    int rc = yyparse(result.ast, result.diagnostics, path);
    std::fclose(fp);

    result.ok = (rc == 0) && !result.diagnostics.hasErrors() && result.ast;
    return result;
}

}  // namespace tsy
