#include "const_eval.h"

using namespace tsy;

namespace tsy::hir {

namespace {

using Result = std::optional<int64_t>;

Result eval(const BaseAST* n, const ConstScope& scope, DiagnosticEngine& diag);

Result evalBinary(const BaseAST* lhs, const BaseAST* rhs, const std::string& op,
                  SourceLocation loc, const ConstScope& scope,
                  DiagnosticEngine& diag) {
    auto l = eval(lhs, scope, diag);
    auto r = eval(rhs, scope, diag);
    if (!l || !r) return std::nullopt;
    if (op == "+") return *l + *r;
    if (op == "-") return *l - *r;
    if (op == "*") return *l * *r;
    if (op == "/") {
        if (*r == 0) {
            diag.error(loc, "division by zero in constant expression");
            return std::nullopt;
        }
        return *l / *r;
    }
    if (op == "%") {
        if (*r == 0) {
            diag.error(loc, "modulo by zero in constant expression");
            return std::nullopt;
        }
        return *l % *r;
    }
    if (op == "==") return *l == *r ? 1 : 0;
    if (op == "!=") return *l != *r ? 1 : 0;
    if (op == "<")  return *l <  *r ? 1 : 0;
    if (op == ">")  return *l >  *r ? 1 : 0;
    if (op == "<=") return *l <= *r ? 1 : 0;
    if (op == ">=") return *l >= *r ? 1 : 0;
    if (op == "&&") return (*l != 0 && *r != 0) ? 1 : 0;
    if (op == "||") return (*l != 0 || *r != 0) ? 1 : 0;
    diag.error(loc, "unsupported operator in constant expression: " + op);
    return std::nullopt;
}

Result eval(const BaseAST* n, const ConstScope& scope, DiagnosticEngine& diag) {
    if (!n) return std::nullopt;

    if (auto* p = dynamic_cast<const ConstExpAST*>(n)) return eval(p->subExp.get(), scope, diag);
    if (auto* p = dynamic_cast<const ExpAST*>(n))       return eval(p->subExp.get(), scope, diag);

    if (auto* p = dynamic_cast<const LOrExpAST*>(n)) {
        if (!p->lOrExp) return eval(p->subExp.get(), scope, diag);
        return evalBinary(p->lOrExp.get(), p->subExp.get(), p->op, p->loc, scope, diag);
    }
    if (auto* p = dynamic_cast<const LAndExpAST*>(n)) {
        if (!p->lAndExp) return eval(p->subExp.get(), scope, diag);
        return evalBinary(p->lAndExp.get(), p->subExp.get(), p->op, p->loc, scope, diag);
    }
    if (auto* p = dynamic_cast<const EqExpAST*>(n)) {
        if (!p->eqExp) return eval(p->subExp.get(), scope, diag);
        return evalBinary(p->eqExp.get(), p->subExp.get(), p->op, p->loc, scope, diag);
    }
    if (auto* p = dynamic_cast<const RelExpAST*>(n)) {
        if (!p->relExp) return eval(p->subExp.get(), scope, diag);
        return evalBinary(p->relExp.get(), p->subExp.get(), p->op, p->loc, scope, diag);
    }
    if (auto* p = dynamic_cast<const AddExpAST*>(n)) {
        if (!p->addExp) return eval(p->subExp.get(), scope, diag);
        return evalBinary(p->addExp.get(), p->subExp.get(), p->op, p->loc, scope, diag);
    }
    if (auto* p = dynamic_cast<const MulExpAST*>(n)) {
        if (!p->mulExp) return eval(p->subExp.get(), scope, diag);
        return evalBinary(p->mulExp.get(), p->subExp.get(), p->op, p->loc, scope, diag);
    }

    if (auto* p = dynamic_cast<const UnaryExpAST*>(n)) {
        if (p->def == UnaryExpAST::def_primaryexp) return eval(p->subExp.get(), scope, diag);
        if (p->def == UnaryExpAST::def_unaryexp) {
            auto v = eval(p->subExp.get(), scope, diag);
            if (!v) return std::nullopt;
            if (p->op == "+") return  *v;
            if (p->op == "-") return -*v;
            if (p->op == "!") return *v == 0 ? 1 : 0;
            diag.error(p->loc, "unsupported unary operator in constant expression: " + p->op);
            return std::nullopt;
        }
        diag.error(p->loc, "function/builtin call is not a constant expression");
        return std::nullopt;
    }

    if (auto* p = dynamic_cast<const PrimaryExpAST*>(n)) {
        switch (p->def) {
            case PrimaryExpAST::def_number:
                return static_cast<int64_t>(p->number);
            case PrimaryExpAST::def_bracketexp:
                return eval(p->subExp.get(), scope, diag);
            case PrimaryExpAST::def_lval: {
                auto it = scope.find(p->lVal);
                if (it == scope.end()) {
                    diag.error(p->loc,
                               "unknown constant '" + p->lVal + "' in expression");
                    return std::nullopt;
                }
                return it->second;
            }
            case PrimaryExpAST::def_array:
                diag.error(p->loc,
                           "array subscript is not a constant expression");
                return std::nullopt;
        }
    }

    return std::nullopt;
}

}  // namespace

std::optional<int64_t> evalConstExp(const BaseAST& node, const ConstScope& scope,
                                    DiagnosticEngine& diag) {
    return eval(&node, scope, diag);
}

ConstScope collectGlobalConstScope(const BaseAST& compUnit, DiagnosticEngine& diag) {
    ConstScope scope;
    auto* cu = dynamic_cast<const CompUnitAST*>(&compUnit);
    if (!cu) return scope;

    for (const auto& d : cu->decls) {
        auto* decl = dynamic_cast<const DeclAST*>(d.get());
        if (!decl || decl->def != DeclAST::def_const) continue;
        auto* cd = dynamic_cast<const ConstDeclAST*>(decl->decl.get());
        if (!cd || cd->bType != "int") continue;
        for (const auto& cdef : cd->constDefs) {
            auto* cdp = dynamic_cast<const ConstDefAST*>(cdef.get());
            if (!cdp || cdp->isArray) continue;
            auto* civ = dynamic_cast<const ConstInitValAST*>(cdp->constInitVal.get());
            if (!civ || civ->isArray || !civ->subExp) continue;
            auto v = evalConstExp(*civ->subExp, scope, diag);
            if (v) scope[cdp->ident] = *v;
        }
    }
    return scope;
}

}  // namespace tsy::hir
