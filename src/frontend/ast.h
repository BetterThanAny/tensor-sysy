#pragma once

// Minimal AST for TensorSysY W0.
//
// Keeps structural parity with sysy-compiler/src/AST.hpp so the migrated
// sysy.l/sysy.y parser can build the tree, but strips every Koopa-IR /
// RISC-V emission hook — those will be replaced by HIR lowering in W2.

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "location.h"

namespace tsy {

class BaseAST;
using BaseASTPtr = std::unique_ptr<BaseAST>;
using MulVecType = std::vector<BaseASTPtr>;

class BaseAST {
   public:
    SourceLocation loc;
    // Kept for grammar compatibility with sysy-compiler's AST.hpp.
    bool isEmptyInitArray = false;
    int32_t arrayDimension = -1;

    virtual ~BaseAST() = default;
    virtual const char* kind() const = 0;
    virtual void dump(std::ostream& os, int indent) const = 0;
};

// --- helpers ---------------------------------------------------------------

void dumpIndent(std::ostream& os, int indent);
void dumpChild(std::ostream& os, const char* label, const BaseASTPtr& child, int indent);
void dumpChildren(std::ostream& os, const char* label, const MulVecType& children, int indent);

// --- top level -------------------------------------------------------------

class CompUnitAST : public BaseAST {
   public:
    MulVecType funcDefs;
    MulVecType decls;

    const char* kind() const override { return "CompUnit"; }
    void dump(std::ostream& os, int indent) const override;
};

class FuncDefAST : public BaseAST {
   public:
    std::string funcType;
    std::string ident;
    BaseASTPtr block;
    MulVecType funcFParams;

    const char* kind() const override { return "FuncDef"; }
    void dump(std::ostream& os, int indent) const override;
};

class FuncFParamAST : public BaseAST {
   public:
    enum Def { def_common, def_array, def_tensor };
    Def def = def_common;
    std::string bType;
    std::string ident;
    MulVecType constExpArray;
    BaseASTPtr tensorType;  // non-null iff def == def_tensor

    const char* kind() const override { return "FuncFParam"; }
    void dump(std::ostream& os, int indent) const override;
};

// Tensor type annotation, e.g. `tensor<f32>[M, N]`.
class TensorTypeAST : public BaseAST {
   public:
    std::string dtype;   // "f32" for now; extended in later weeks.
    MulVecType dims;     // each element is a ConstExpAST.

    const char* kind() const override { return "TensorType"; }
    void dump(std::ostream& os, int indent) const override;
};

// --- decls ------------------------------------------------------------------

class DeclAST : public BaseAST {
   public:
    enum Def { def_const, def_var };
    Def def = def_var;
    BaseASTPtr decl;

    const char* kind() const override { return "Decl"; }
    void dump(std::ostream& os, int indent) const override;
};

class ConstDeclAST : public BaseAST {
   public:
    std::string bType;
    MulVecType constDefs;

    const char* kind() const override { return "ConstDecl"; }
    void dump(std::ostream& os, int indent) const override;
};

class ConstDefAST : public BaseAST {
   public:
    std::string ident;
    BaseASTPtr constInitVal;
    int isArray = 0;
    MulVecType constExpArray;

    const char* kind() const override { return "ConstDef"; }
    void dump(std::ostream& os, int indent) const override;
};

class ConstInitValAST : public BaseAST {
   public:
    BaseASTPtr subExp;
    int isArray = 0;
    MulVecType constInitVals;

    const char* kind() const override { return "ConstInitVal"; }
    void dump(std::ostream& os, int indent) const override;
};

class VarDeclAST : public BaseAST {
   public:
    std::string bType;
    MulVecType varDefs;

    const char* kind() const override { return "VarDecl"; }
    void dump(std::ostream& os, int indent) const override;
};

class VarDefAST : public BaseAST {
   public:
    std::string ident;
    BaseASTPtr initVal;
    MulVecType constExpArray;
    int isInitialized = 0;
    BaseASTPtr tensorType;  // non-null for `tensor<...>[...] name = ...` forms

    const char* kind() const override { return "VarDef"; }
    void dump(std::ostream& os, int indent) const override;
};

class InitValAST : public BaseAST {
   public:
    BaseASTPtr subExp;
    int isArray = 0;
    MulVecType initVals;

    const char* kind() const override { return "InitVal"; }
    void dump(std::ostream& os, int indent) const override;
};

// --- statements -------------------------------------------------------------

class BlockAST : public BaseAST {
   public:
    MulVecType blockItems;
    std::string func;  // set by parser for outer function blocks.

    const char* kind() const override { return "Block"; }
    void dump(std::ostream& os, int indent) const override;
};

class BlockItemAST : public BaseAST {
   public:
    enum Def { def_decl, def_stmt };
    Def def = def_stmt;
    BaseASTPtr blockItem;

    const char* kind() const override { return "BlockItem"; }
    void dump(std::ostream& os, int indent) const override;
};

class ComplexStmtAST : public BaseAST {
   public:
    enum Def { def_simple, def_ifelse, def_while, def_openif };
    Def def = def_simple;
    BaseASTPtr subExp;
    BaseASTPtr subStmt;
    BaseASTPtr elseStmt;

    const char* kind() const override { return "ComplexStmt"; }
    void dump(std::ostream& os, int indent) const override;
};

class StmtAST : public BaseAST {
   public:
    enum Def { def_ret, def_lval, def_exp, def_block, def_break, def_continue, def_array };
    Def def = def_exp;
    std::string lVal;
    BaseASTPtr subExp;
    MulVecType expArray;

    const char* kind() const override { return "Stmt"; }
    void dump(std::ostream& os, int indent) const override;
};

// --- expressions ------------------------------------------------------------

class ConstExpAST : public BaseAST {
   public:
    BaseASTPtr subExp;
    const char* kind() const override { return "ConstExp"; }
    void dump(std::ostream& os, int indent) const override;
};

class ExpAST : public BaseAST {
   public:
    BaseASTPtr subExp;
    const char* kind() const override { return "Exp"; }
    void dump(std::ostream& os, int indent) const override;
};

class PrimaryExpAST : public BaseAST {
   public:
    enum Def { def_bracketexp, def_lval, def_number, def_array };
    Def def = def_number;
    BaseASTPtr subExp;
    std::string lVal;
    int32_t number = 0;
    std::string arrayIdent;
    MulVecType expArray;

    const char* kind() const override { return "PrimaryExp"; }
    void dump(std::ostream& os, int indent) const override;
};

class UnaryExpAST : public BaseAST {
   public:
    enum Def { def_primaryexp, def_unaryexp, def_func, def_builtin };
    Def def = def_primaryexp;
    BaseASTPtr subExp;
    std::string op;
    std::string ident;  // function name for def_func; builtin name (without '@')
                        // for def_builtin.
    MulVecType funcRParams;

    const char* kind() const override { return "UnaryExp"; }
    void dump(std::ostream& os, int indent) const override;
};

class MulExpAST : public BaseAST {
   public:
    BaseASTPtr mulExp;
    BaseASTPtr subExp;
    std::string op;
    const char* kind() const override { return "MulExp"; }
    void dump(std::ostream& os, int indent) const override;
};

class AddExpAST : public BaseAST {
   public:
    BaseASTPtr addExp;
    BaseASTPtr subExp;
    std::string op;
    const char* kind() const override { return "AddExp"; }
    void dump(std::ostream& os, int indent) const override;
};

class RelExpAST : public BaseAST {
   public:
    BaseASTPtr relExp;
    BaseASTPtr subExp;
    std::string op;
    const char* kind() const override { return "RelExp"; }
    void dump(std::ostream& os, int indent) const override;
};

class EqExpAST : public BaseAST {
   public:
    BaseASTPtr eqExp;
    BaseASTPtr subExp;
    std::string op;
    const char* kind() const override { return "EqExp"; }
    void dump(std::ostream& os, int indent) const override;
};

class LAndExpAST : public BaseAST {
   public:
    BaseASTPtr lAndExp;
    BaseASTPtr subExp;
    std::string op;
    const char* kind() const override { return "LAndExp"; }
    void dump(std::ostream& os, int indent) const override;
};

class LOrExpAST : public BaseAST {
   public:
    BaseASTPtr lOrExp;
    BaseASTPtr subExp;
    std::string op;
    const char* kind() const override { return "LOrExp"; }
    void dump(std::ostream& os, int indent) const override;
};

}  // namespace tsy
