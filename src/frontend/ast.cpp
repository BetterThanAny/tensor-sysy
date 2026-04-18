#include "ast.h"

namespace tsy {

void dumpIndent(std::ostream& os, int indent) {
    for (int i = 0; i < indent; ++i) os.put(' ');
}

void dumpChild(std::ostream& os, const char* label, const BaseASTPtr& child, int indent) {
    if (!child) return;
    dumpIndent(os, indent);
    os << label << ":\n";
    child->dump(os, indent + 2);
}

void dumpChildren(std::ostream& os, const char* label, const MulVecType& children, int indent) {
    if (children.empty()) return;
    dumpIndent(os, indent);
    os << label << ":\n";
    for (const auto& c : children) {
        if (c) c->dump(os, indent + 2);
    }
}

static void header(std::ostream& os, int indent, const BaseAST& n, const std::string& extra = {}) {
    dumpIndent(os, indent);
    os << n.kind();
    if (n.loc.valid()) os << " @" << n.loc.line << ":" << n.loc.column;
    if (!extra.empty()) os << " " << extra;
    os << "\n";
}

// --- top level -------------------------------------------------------------

void CompUnitAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this);
    dumpChildren(os, "decls", decls, indent + 2);
    dumpChildren(os, "funcDefs", funcDefs, indent + 2);
}

void FuncDefAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, funcType + " " + ident);
    dumpChildren(os, "params", funcFParams, indent + 2);
    dumpChild(os, "body", block, indent + 2);
}

void FuncFParamAST::dump(std::ostream& os, int indent) const {
    const char* d = "scalar";
    switch (def) {
        case def_common: d = "scalar"; break;
        case def_array: d = "array"; break;
        case def_tensor: d = "tensor"; break;
    }
    std::string extra = std::string(d) + " ";
    if (def == def_tensor) {
        extra += ident;
    } else {
        extra += bType + " " + ident;
    }
    header(os, indent, *this, extra);
    dumpChildren(os, "dims", constExpArray, indent + 2);
    dumpChild(os, "tensorType", tensorType, indent + 2);
}

void TensorTypeAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, dtype);
    dumpChildren(os, "dims", dims, indent + 2);
}

// --- decls -----------------------------------------------------------------

void DeclAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, def == def_const ? "const" : "var");
    dumpChild(os, "decl", decl, indent + 2);
}

void ConstDeclAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, bType);
    dumpChildren(os, "defs", constDefs, indent + 2);
}

void ConstDefAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, ident + (isArray ? " [array]" : ""));
    dumpChildren(os, "dims", constExpArray, indent + 2);
    dumpChild(os, "init", constInitVal, indent + 2);
}

void ConstInitValAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, isArray ? "{...}" : "scalar");
    dumpChild(os, "exp", subExp, indent + 2);
    dumpChildren(os, "elems", constInitVals, indent + 2);
}

void VarDeclAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, bType);
    dumpChildren(os, "defs", varDefs, indent + 2);
}

void VarDefAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, ident + (isInitialized ? " =" : ""));
    dumpChildren(os, "dims", constExpArray, indent + 2);
    dumpChild(os, "tensorType", tensorType, indent + 2);
    dumpChild(os, "init", initVal, indent + 2);
}

void InitValAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, isArray ? "{...}" : "scalar");
    dumpChild(os, "exp", subExp, indent + 2);
    dumpChildren(os, "elems", initVals, indent + 2);
}

// --- statements ------------------------------------------------------------

void BlockAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, func);
    dumpChildren(os, "items", blockItems, indent + 2);
}

void BlockItemAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this, def == def_decl ? "decl" : "stmt");
    dumpChild(os, "item", blockItem, indent + 2);
}

void ComplexStmtAST::dump(std::ostream& os, int indent) const {
    const char* tag = "simple";
    switch (def) {
        case def_simple: tag = "simple"; break;
        case def_ifelse: tag = "if-else"; break;
        case def_while: tag = "while"; break;
        case def_openif: tag = "open-if"; break;
    }
    header(os, indent, *this, tag);
    dumpChild(os, "cond", subExp, indent + 2);
    dumpChild(os, "then", subStmt, indent + 2);
    dumpChild(os, "else", elseStmt, indent + 2);
}

void StmtAST::dump(std::ostream& os, int indent) const {
    const char* tag = "exp";
    switch (def) {
        case def_ret: tag = "return"; break;
        case def_lval: tag = "assign"; break;
        case def_exp: tag = "exp"; break;
        case def_block: tag = "block"; break;
        case def_break: tag = "break"; break;
        case def_continue: tag = "continue"; break;
        case def_array: tag = "assign-array"; break;
    }
    std::string extra = tag;
    if (!lVal.empty()) extra += " " + lVal;
    header(os, indent, *this, extra);
    dumpChildren(os, "idx", expArray, indent + 2);
    dumpChild(os, "exp", subExp, indent + 2);
}

// --- expressions -----------------------------------------------------------

void ConstExpAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this);
    dumpChild(os, "exp", subExp, indent + 2);
}

void ExpAST::dump(std::ostream& os, int indent) const {
    header(os, indent, *this);
    dumpChild(os, "exp", subExp, indent + 2);
}

void PrimaryExpAST::dump(std::ostream& os, int indent) const {
    const char* tag = "number";
    std::string extra;
    switch (def) {
        case def_bracketexp: tag = "paren"; break;
        case def_lval: tag = "lval"; extra = lVal; break;
        case def_number: tag = "number"; extra = std::to_string(number); break;
        case def_array: tag = "array"; extra = arrayIdent; break;
    }
    std::string line = tag;
    if (!extra.empty()) line += " " + extra;
    header(os, indent, *this, line);
    dumpChildren(os, "idx", expArray, indent + 2);
    dumpChild(os, "inner", subExp, indent + 2);
}

void UnaryExpAST::dump(std::ostream& os, int indent) const {
    const char* tag = "primary";
    switch (def) {
        case def_primaryexp: tag = "primary"; break;
        case def_unaryexp: tag = "unary"; break;
        case def_func: tag = "call"; break;
        case def_builtin: tag = "builtin"; break;
    }
    std::string extra = tag;
    if (!op.empty()) extra += " op=" + op;
    if (!ident.empty()) {
        extra += " ";
        if (def == def_builtin) extra += "@";
        extra += ident;
    }
    header(os, indent, *this, extra);
    dumpChildren(os, "args", funcRParams, indent + 2);
    dumpChild(os, "sub", subExp, indent + 2);
}

void MulExpAST::dump(std::ostream& os, int indent) const {
    std::string extra = op.empty() ? std::string("leaf") : op;
    header(os, indent, *this, extra);
    dumpChild(os, "lhs", mulExp, indent + 2);
    dumpChild(os, "rhs", subExp, indent + 2);
}

void AddExpAST::dump(std::ostream& os, int indent) const {
    std::string extra = op.empty() ? std::string("leaf") : op;
    header(os, indent, *this, extra);
    dumpChild(os, "lhs", addExp, indent + 2);
    dumpChild(os, "rhs", subExp, indent + 2);
}

void RelExpAST::dump(std::ostream& os, int indent) const {
    std::string extra = op.empty() ? std::string("leaf") : op;
    header(os, indent, *this, extra);
    dumpChild(os, "lhs", relExp, indent + 2);
    dumpChild(os, "rhs", subExp, indent + 2);
}

void EqExpAST::dump(std::ostream& os, int indent) const {
    std::string extra = op.empty() ? std::string("leaf") : op;
    header(os, indent, *this, extra);
    dumpChild(os, "lhs", eqExp, indent + 2);
    dumpChild(os, "rhs", subExp, indent + 2);
}

void LAndExpAST::dump(std::ostream& os, int indent) const {
    std::string extra = op.empty() ? std::string("leaf") : op;
    header(os, indent, *this, extra);
    dumpChild(os, "lhs", lAndExp, indent + 2);
    dumpChild(os, "rhs", subExp, indent + 2);
}

void LOrExpAST::dump(std::ostream& os, int indent) const {
    std::string extra = op.empty() ? std::string("leaf") : op;
    header(os, indent, *this, extra);
    dumpChild(os, "lhs", lOrExp, indent + 2);
    dumpChild(os, "rhs", subExp, indent + 2);
}

}  // namespace tsy
