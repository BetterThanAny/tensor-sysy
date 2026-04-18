#include "lowering.h"

#include <string>
#include <unordered_map>

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
    ConstScope constScope;
    std::unique_ptr<Module> mod = std::make_unique<Module>();

    // Per-function state reset in lowerFuncDef.
    Function* curFn = nullptr;
    std::unordered_map<std::string, ValuePtr> names;

    explicit Lowerer(DiagnosticEngine& d) : diag(d) {}

    TensorType convertTensorType(const TensorTypeAST& t) {
        TensorType out;
        out.dtype = DType::F32;  // only dtype for W2/W3.
        for (const auto& d : t.dims) {
            Dim dim;
            dim.symbol = symbolizeConstExp(d.get());
            // Route through the const evaluator so names like `M`, `N` get a
            // concrete int64 whenever they resolve against the global const
            // scope. Diagnostics from the evaluator surface here.
            auto resolved = evalConstExp(*d, constScope, diag);
            if (resolved) dim.resolved = *resolved;
            out.shape.dims.push_back(std::move(dim));
        }
        return out;
    }

    void run(const CompUnitAST& cu) {
        constScope = collectGlobalConstScope(cu, diag);
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
        names.clear();

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
            names[param->ident] = v;
        }

        if (auto* blk = dynamic_cast<const BlockAST*>(fn.block.get())) {
            lowerBlock(*blk);
        }

        mod->funcs.push_back(std::move(f));
        curFn = nullptr;
    }

    void lowerBlock(const BlockAST& blk) {
        for (const auto& item : blk.blockItems) {
            auto* bi = dynamic_cast<const BlockItemAST*>(item.get());
            if (!bi) continue;
            if (bi->def == BlockItemAST::def_decl) {
                auto* d = dynamic_cast<const DeclAST*>(bi->blockItem.get());
                if (d && d->def == DeclAST::def_var) {
                    if (auto* vd = dynamic_cast<const VarDeclAST*>(d->decl.get())) {
                        lowerVarDecl(*vd);
                    }
                }
            } else {
                if (auto* cs = dynamic_cast<const ComplexStmtAST*>(bi->blockItem.get())) {
                    lowerStmt(*cs);
                }
            }
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

        auto result = std::make_shared<Value>();
        result->name = "%" + vdef.ident;
        result->type = type;
        names[vdef.ident] = result;

        if (!vdef.initVal) return;  // uninitialised tensor; no op in W2.

        auto* iv = dynamic_cast<const InitValAST*>(vdef.initVal.get());
        if (!iv || iv->isArray || !iv->subExp) {
            emitUnknown(result, "tensor-array-init", vdef.loc);
            return;
        }

        const UnaryExpAST* u = unwrapToUnary(iv->subExp.get());
        if (!u) {
            emitUnknown(result, "complex-init", vdef.loc);
            return;
        }
        if (u->def != UnaryExpAST::def_builtin) {
            const char* why = u->def == UnaryExpAST::def_func ? "call-init" : "non-builtin-init";
            emitUnknown(result, why, vdef.loc);
            return;
        }
        emitBuiltinOp(result, *u);
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
        auto it = names.find(pe->lVal);
        if (it == names.end()) {
            diag.error(arg->loc, "unresolved tensor value '" + pe->lVal + "'");
            return nullptr;
        }
        return it->second;
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

    void lowerStmt(const ComplexStmtAST& cs) {
        if (cs.def != ComplexStmtAST::def_simple || !cs.subExp) return;
        auto* stmt = dynamic_cast<const StmtAST*>(cs.subExp.get());
        if (!stmt) return;
        if (stmt->def == StmtAST::def_ret) {
            auto op = std::make_unique<Op>();
            op->kind = OpKind::Return;
            op->loc = stmt->loc;
            curFn->ops.push_back(std::move(op));
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
