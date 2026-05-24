#include "lowering.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "const_eval.h"

using namespace tsy;

namespace tsy::hir {

namespace {

// Walk through the expression chain AST (ExpAST → LOrExp → ... → UnaryExp)
// and return the leaf UnaryExp *iff* every intermediate level is a pure
// pass-through (no binary operator applied at any rung). This is enough for
// W2, where real codegen of arithmetic expressions is W7 territory.
const UnaryExpAST* unwrapToUnary(const BaseAST* n) {
    if (!n) return nullptr;
    if (auto* p = dynamic_cast<const ConstExpAST*>(n)) return unwrapToUnary(p->subExp.get());
    if (auto* p = dynamic_cast<const ExpAST*>(n))       return unwrapToUnary(p->subExp.get());
    if (auto* p = dynamic_cast<const LOrExpAST*>(n)) {
        if (p->lOrExp || !p->op.empty()) return nullptr;
        return unwrapToUnary(p->subExp.get());
    }
    if (auto* p = dynamic_cast<const LAndExpAST*>(n)) {
        if (p->lAndExp || !p->op.empty()) return nullptr;
        return unwrapToUnary(p->subExp.get());
    }
    if (auto* p = dynamic_cast<const EqExpAST*>(n)) {
        if (p->eqExp || !p->op.empty()) return nullptr;
        return unwrapToUnary(p->subExp.get());
    }
    if (auto* p = dynamic_cast<const RelExpAST*>(n)) {
        if (p->relExp || !p->op.empty()) return nullptr;
        return unwrapToUnary(p->subExp.get());
    }
    if (auto* p = dynamic_cast<const AddExpAST*>(n)) {
        if (p->addExp || !p->op.empty()) return nullptr;
        return unwrapToUnary(p->subExp.get());
    }
    if (auto* p = dynamic_cast<const MulExpAST*>(n)) {
        if (p->mulExp || !p->op.empty()) return nullptr;
        return unwrapToUnary(p->subExp.get());
    }
    if (auto* p = dynamic_cast<const UnaryExpAST*>(n)) return p;
    return nullptr;
}

// Produce the source-level symbol for a ConstExp used in a tensor dim.
// Recognises single numeric literals and single identifier references.
// Anything more complex falls back to "?" and leaves the work to W3's const
// evaluator, which will replace it with a resolved integer.
std::string symbolizeConstExp(const BaseAST* n) {
    const UnaryExpAST* u = unwrapToUnary(n);
    if (!u || u->def != UnaryExpAST::def_primaryexp) return "?";
    const auto* p = dynamic_cast<const PrimaryExpAST*>(u->subExp.get());
    if (!p) return "?";
    if (p->def == PrimaryExpAST::def_number) return std::to_string(p->number);
    if (p->def == PrimaryExpAST::def_lval) return p->lVal;
    return "?";
}

struct Lowerer {
    DiagnosticEngine& diag;
    std::vector<ConstScope> constScopes;
    std::unique_ptr<Module> mod = std::make_unique<Module>();

    // Per-function state reset in lowerFuncDef.
    Function* curFn = nullptr;
    std::vector<std::unordered_map<std::string, ValuePtr>> nameScopes;
    bool terminated = false;

    explicit Lowerer(DiagnosticEngine& d) : diag(d) {}

    ConstScope visibleConstScope() const {
        ConstScope merged;
        for (const auto& scope : constScopes) {
            for (const auto& kv : scope) merged[kv.first] = kv.second;
        }
        return merged;
    }

    ValuePtr resolveName(const std::string& name) const {
        for (auto it = nameScopes.rbegin(); it != nameScopes.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) return found->second;
        }
        return nullptr;
    }

    bool nameExistsVisible(const std::string& name) const {
        return static_cast<bool>(resolveName(name));
    }

    bool defineName(const std::string& name, ValuePtr value, SourceLocation loc,
                    const char* what) {
        if (nameExistsVisible(name)) {
            diag.error(loc, std::string("duplicate ") + what + " '" + name + "'");
            return false;
        }
        if (nameScopes.empty()) nameScopes.push_back({});
        nameScopes.back()[name] = std::move(value);
        return true;
    }

    TensorType convertTensorType(const TensorTypeAST& t) {
        TensorType out;
        out.dtype = DType::F32;  // only dtype for W2/W3.
        ConstScope scope = visibleConstScope();
        for (const auto& d : t.dims) {
            Dim dim;
            dim.symbol = symbolizeConstExp(d.get());
            // Route through the const evaluator so names like `M`, `N` get a
            // concrete int64 whenever they resolve against the visible const
            // scope. Diagnostics from the evaluator surface here.
            auto resolved = evalConstExp(*d, scope, diag);
            if (resolved) dim.resolved = *resolved;
            out.shape.dims.push_back(std::move(dim));
        }
        return out;
    }

    void run(const CompUnitAST& cu) {
        constScopes.clear();
        constScopes.push_back(collectGlobalConstScope(cu, diag));
        for (const auto& fn : cu.funcDefs) {
            auto* fd = dynamic_cast<const FuncDefAST*>(fn.get());
            if (fd) lowerFuncDef(*fd);
        }
    }

    void lowerFuncDef(const FuncDefAST& fn) {
        auto f = std::make_unique<Function>();
        f->name = fn.ident;
        f->return_type = fn.funcType;
        f->loc = fn.loc;
        curFn = f.get();
        nameScopes.clear();
        nameScopes.push_back({});
        terminated = false;

        for (const auto& p : fn.funcFParams) {
            auto* param = dynamic_cast<const FuncFParamAST*>(p.get());
            if (!param) continue;
            auto v = std::make_shared<Value>();
            v->name = "%" + param->ident;
            if (param->def == FuncFParamAST::def_tensor && param->tensorType) {
                auto* tt = dynamic_cast<const TensorTypeAST*>(param->tensorType.get());
                if (tt) v->type = convertTensorType(*tt);
            }
            curFn->params.push_back(v);
            defineName(param->ident, v, param->loc, "value name");
        }

        if (auto* blk = dynamic_cast<const BlockAST*>(fn.block.get())) {
            lowerBlock(*blk);
        }

        mod->funcs.push_back(std::move(f));
        curFn = nullptr;
        nameScopes.clear();
        terminated = false;
    }

    void lowerBlock(const BlockAST& blk) {
        constScopes.push_back({});
        nameScopes.push_back({});
        for (const auto& item : blk.blockItems) {
            if (terminated) break;
            auto* bi = dynamic_cast<const BlockItemAST*>(item.get());
            if (!bi) continue;
            if (bi->def == BlockItemAST::def_decl) {
                auto* d = dynamic_cast<const DeclAST*>(bi->blockItem.get());
                if (d) {
                    if (d->def == DeclAST::def_const) {
                        if (auto* cd = dynamic_cast<const ConstDeclAST*>(d->decl.get())) {
                            lowerConstDecl(*cd);
                        }
                    } else if (d->def == DeclAST::def_var) {
                        if (auto* vd = dynamic_cast<const VarDeclAST*>(d->decl.get())) {
                            lowerVarDecl(*vd);
                        }
                    }
                }
            } else {
                if (auto* cs = dynamic_cast<const ComplexStmtAST*>(bi->blockItem.get())) {
                    lowerStmt(*cs);
                }
            }
        }
        nameScopes.pop_back();
        constScopes.pop_back();
    }

    void lowerConstDecl(const ConstDeclAST& cd) {
        if (cd.bType != "int" || constScopes.empty()) return;
        for (const auto& cdef : cd.constDefs) {
            auto* c = dynamic_cast<const ConstDefAST*>(cdef.get());
            if (!c || c->isArray) continue;
            auto* init = dynamic_cast<const ConstInitValAST*>(c->constInitVal.get());
            if (!init || init->isArray || !init->subExp) continue;
            auto v = evalConstExp(*init->subExp, visibleConstScope(), diag);
            if (v) constScopes.back()[c->ident] = *v;
        }
    }

    void lowerVarDecl(const VarDeclAST& vd) {
        for (const auto& vdef : vd.varDefs) {
            auto* v = dynamic_cast<const VarDefAST*>(vdef.get());
            if (v && v->tensorType) lowerTensorVarDef(*v);
            // Non-tensor locals are intentionally invisible to HIR in W2.
        }
    }

    void lowerTensorVarDef(const VarDefAST& vdef) {
        auto* tt = dynamic_cast<const TensorTypeAST*>(vdef.tensorType.get());
        TensorType type = tt ? convertTensorType(*tt) : TensorType{};

        if (nameExistsVisible(vdef.ident)) {
            diag.error(vdef.loc, "duplicate tensor value '" + vdef.ident + "'");
            return;
        }

        auto result = std::make_shared<Value>();
        result->name = "%" + vdef.ident;
        result->type = type;

        if (!vdef.initVal) {
            defineName(vdef.ident, result, vdef.loc, "tensor value");
            return;
        }

        auto* iv = dynamic_cast<const InitValAST*>(vdef.initVal.get());
        if (!iv || iv->isArray || !iv->subExp) {
            emitUnknown(result, "tensor-array-init", vdef.loc);
            defineName(vdef.ident, result, vdef.loc, "tensor value");
            return;
        }

        const UnaryExpAST* u = unwrapToUnary(iv->subExp.get());
        if (!u) {
            emitUnknown(result, "complex-init", vdef.loc);
            defineName(vdef.ident, result, vdef.loc, "tensor value");
            return;
        }
        if (u->def != UnaryExpAST::def_builtin) {
            const char* why = u->def == UnaryExpAST::def_func ? "call-init" : "non-builtin-init";
            emitUnknown(result, why, vdef.loc);
            defineName(vdef.ident, result, vdef.loc, "tensor value");
            return;
        }
        emitBuiltinOp(result, *u);
        defineName(vdef.ident, result, vdef.loc, "tensor value");
    }

    void emitBuiltinOp(ValuePtr result, const UnaryExpAST& u) {
        auto op = std::make_unique<Op>();
        op->kind = builtinKindFromName(u.ident);
        op->builtin_name = u.ident;
        op->loc = u.loc;
        op->results.push_back(result);
        result->defining_op = op.get();

        for (const auto& arg : u.funcRParams) {
            if (auto v = resolveArg(arg.get())) {
                op->operands.push_back(v);
            }
            // Unresolvable args are left out; W3's verifier surfaces the arity
            // problem with a dedicated message.
        }
        curFn->ops.push_back(std::move(op));
    }

    ValuePtr resolveArg(const BaseAST* arg) {
        const UnaryExpAST* u = unwrapToUnary(arg);
        if (!u || u->def != UnaryExpAST::def_primaryexp) return nullptr;
        auto* pe = dynamic_cast<const PrimaryExpAST*>(u->subExp.get());
        if (!pe || pe->def != PrimaryExpAST::def_lval) return nullptr;
        auto value = resolveName(pe->lVal);
        if (!value) {
            diag.error(arg->loc, "unresolved tensor value '" + pe->lVal + "'");
            return nullptr;
        }
        return value;
    }

    void emitUnknown(ValuePtr result, const std::string& reason, SourceLocation loc) {
        auto op = std::make_unique<Op>();
        op->kind = OpKind::Unknown;
        op->builtin_name = reason;
        op->loc = loc;
        op->results.push_back(result);
        result->defining_op = op.get();
        curFn->ops.push_back(std::move(op));
    }

    bool containsTensorEffect(const BaseAST* n) const {
        if (!n) return false;
        if (auto* b = dynamic_cast<const BlockAST*>(n)) {
            for (const auto& item : b->blockItems) {
                if (containsTensorEffect(item.get())) return true;
            }
            return false;
        }
        if (auto* bi = dynamic_cast<const BlockItemAST*>(n)) {
            return containsTensorEffect(bi->blockItem.get());
        }
        if (auto* d = dynamic_cast<const DeclAST*>(n)) {
            if (d->def != DeclAST::def_var) return false;
            auto* vd = dynamic_cast<const VarDeclAST*>(d->decl.get());
            if (!vd) return false;
            for (const auto& vdef : vd->varDefs) {
                auto* v = dynamic_cast<const VarDefAST*>(vdef.get());
                if (v && v->tensorType) return true;
            }
            return false;
        }
        if (auto* cs = dynamic_cast<const ComplexStmtAST*>(n)) {
            if (containsTensorEffect(cs->subExp.get())) return true;
            if (containsTensorEffect(cs->subStmt.get())) return true;
            if (containsTensorEffect(cs->elseStmt.get())) return true;
            return false;
        }
        if (auto* stmt = dynamic_cast<const StmtAST*>(n)) {
            if (stmt->def == StmtAST::def_block) {
                return containsTensorEffect(stmt->subExp.get());
            }
            if (stmt->def == StmtAST::def_lval) {
                return nameExistsVisible(stmt->lVal);
            }
            return false;
        }
        return false;
    }

    void lowerStmt(const ComplexStmtAST& cs) {
        if (cs.def != ComplexStmtAST::def_simple) {
            if (containsTensorEffect(cs.subStmt.get()) ||
                containsTensorEffect(cs.elseStmt.get())) {
                diag.error(cs.loc,
                           "tensor operations inside control-flow statements are "
                           "not supported by HIR lowering");
            }
            return;
        }
        if (!cs.subExp) return;
        auto* stmt = dynamic_cast<const StmtAST*>(cs.subExp.get());
        if (!stmt) return;
        if (stmt->def == StmtAST::def_block) {
            if (auto* block = dynamic_cast<const BlockAST*>(stmt->subExp.get())) {
                lowerBlock(*block);
            }
            return;
        }
        if (stmt->def == StmtAST::def_ret) {
            auto op = std::make_unique<Op>();
            op->kind = OpKind::Return;
            op->loc = stmt->loc;
            curFn->ops.push_back(std::move(op));
            terminated = true;
        } else if (stmt->def == StmtAST::def_lval) {
            if (nameExistsVisible(stmt->lVal)) {
                diag.error(stmt->loc,
                           "tensor assignment to '" + stmt->lVal +
                           "' is not supported; initialize tensors at declaration");
            }
        }
    }
};

}  // namespace

std::unique_ptr<Module> lowerAstToHir(const tsy::BaseAST& compUnit,
                                      DiagnosticEngine& diag) {
    Lowerer l(diag);
    if (auto* cu = dynamic_cast<const CompUnitAST*>(&compUnit)) {
        l.run(*cu);
    }
    return std::move(l.mod);
}

}  // namespace tsy::hir
