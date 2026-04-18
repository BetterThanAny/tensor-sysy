%code requires {
// Emitted into both sysy.tab.hpp and sysy.tab.cpp, so the yyparse()
// signature (which names tsy::BaseASTPtr / tsy::DiagnosticEngine via
// %parse-param) is well-formed wherever the header is included.
#include "ast.h"
#include "diagnostics.h"
}

%{

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ast.h"
#include "diagnostics.h"

using namespace std;
using namespace tsy;

int yylex();

// Forward-declare yyerror so bison 2.3's generated yyparse body can call it
// before its definition at the end of this file. Signature must match the
// %parse-param list below plus the trailing message.
void yyerror(tsy::BaseASTPtr &ast, tsy::DiagnosticEngine &diag,
             const std::string &filename, const char *s);

// YYLTYPE is defined by bison (and exposed through sysy.tab.hpp when
// %locations + --defines are set). Bison 2.3 emits YYSTYPE/YYLTYPE *after*
// this prologue, so LOC() is intentionally a macro rather than a function
// — it only needs the type at expansion time inside action code.
#define LOC(L) tsy::SourceLocation{filename, (L).first_line, (L).first_column}

%}

%locations
%parse-param { tsy::BaseASTPtr &ast }
%parse-param { tsy::DiagnosticEngine &diag }
%parse-param { const std::string &filename }

%union {
  std::string *str_val;
  int int_val;
  tsy::BaseAST *ast_val;
  tsy::MulVecType *mul_val;
}

%token INT RETURN CONST VOID IF ELSE WHILE BREAK CONTINUE
%token TENSOR F32 AT
%token <str_val> IDENT UNARYOP MULOP ADDOP RELOP EQOP LANDOP LOROP
%token <int_val> INT_CONST

%type <ast_val> FuncDef Block BlockItem Stmt ComplexStmt OpenStmt ClosedStmt
%type <ast_val> Decl ConstDecl ConstDef ConstInitVal VarDecl VarDef InitVal
%type <ast_val> Exp ConstExp PrimaryExp UnaryExp MulExp AddExp RelExp EqExp LAndExp LOrExp
%type <ast_val> TensorType
%type <mul_val> BlockItems ConstDefs VarDefs FuncFParams FuncRParams
%type <mul_val> InitVals ConstInitVals ExpArray ConstExpArray TensorDims
%type <ast_val> CompUnitList FuncFParam
%type <int_val> Number
%type <str_val> Type LVal

%%

CompUnit
  : CompUnitList {
    auto comp_unit = BaseASTPtr($1);
    ast = std::move(comp_unit);
  }
  ;

CompUnitList
  : FuncDef {
    auto comp_unit = new CompUnitAST();
    comp_unit->loc = LOC(@1);
    auto func_def = BaseASTPtr($1);
    comp_unit->funcDefs.push_back(std::move(func_def));
    $$ = comp_unit;
  }
  | Decl {
    auto comp_unit = new CompUnitAST();
    comp_unit->loc = LOC(@1);
    auto decl = BaseASTPtr($1);
    comp_unit->decls.push_back(std::move(decl));
    $$ = comp_unit;
  }
  | CompUnitList FuncDef {
    auto comp_unit = (CompUnitAST*)($1);
    auto func_def = BaseASTPtr($2);
    comp_unit->funcDefs.push_back(std::move(func_def));
    $$ = comp_unit;
  }
  | CompUnitList Decl {
    auto comp_unit = (CompUnitAST*)($1);
    auto decl = BaseASTPtr($2);
    comp_unit->decls.push_back(std::move(decl));
    $$ = comp_unit;
  }
  ;

FuncDef
  : Type IDENT '(' ')' Block {
    auto func_def = new FuncDefAST();
    func_def->loc = LOC(@1);
    func_def->funcType = *unique_ptr<string>($1);
    func_def->ident = *unique_ptr<string>($2);
    func_def->block = BaseASTPtr($5);
    $$ = func_def;
  }
  | Type IDENT '(' FuncFParams ')' Block {
    auto func_def = new FuncDefAST();
    func_def->loc = LOC(@1);
    func_def->funcType = *unique_ptr<string>($1);
    func_def->ident = *unique_ptr<string>($2);
    MulVecType *vec = ($4);
    for (auto it = vec->begin(); it != vec->end(); it++)
        func_def->funcFParams.push_back(std::move(*it));
    delete vec;
    func_def->block = BaseASTPtr($6);
    ((BlockAST*)(func_def->block).get())->func = func_def->ident;
    $$ = func_def;
  }
  ;

FuncFParams
  : FuncFParam {
    MulVecType *vec = new MulVecType;
    vec->push_back(BaseASTPtr($1));
    $$ = vec;
  }
  | FuncFParams ',' FuncFParam {
    MulVecType *vec = ($1);
    vec->push_back(BaseASTPtr($3));
    $$ = vec;
  }
  ;

FuncFParam
  : Type IDENT {
    auto p = new FuncFParamAST();
    p->loc = LOC(@1);
    p->def = FuncFParamAST::def_common;
    p->bType = *unique_ptr<string>($1);
    p->ident = *unique_ptr<string>($2);
    $$ = p;
  }
  | Type IDENT '[' ']' {
    auto p = new FuncFParamAST();
    p->loc = LOC(@1);
    p->def = FuncFParamAST::def_array;
    p->bType = *unique_ptr<string>($1);
    p->ident = *unique_ptr<string>($2);
    p->arrayDimension = 1;
    $$ = p;
  }
  | Type IDENT '[' ']' ConstExpArray {
    auto p = new FuncFParamAST();
    p->loc = LOC(@1);
    p->def = FuncFParamAST::def_array;
    p->bType = *unique_ptr<string>($1);
    p->ident = *unique_ptr<string>($2);
    MulVecType *vec = ($5);
    for (auto it = vec->begin(); it != vec->end(); it++)
        p->constExpArray.push_back(std::move(*it));
    delete vec;
    p->arrayDimension = (int)p->constExpArray.size() + 1;
    $$ = p;
  }
  | TensorType IDENT {
    auto p = new FuncFParamAST();
    p->loc = LOC(@1);
    p->def = FuncFParamAST::def_tensor;
    p->tensorType = BaseASTPtr($1);
    p->ident = *unique_ptr<string>($2);
    $$ = p;
  }
  ;

TensorType
  : TENSOR '<' F32 '>' '[' TensorDims ']' {
    auto t = new TensorTypeAST();
    t->loc = LOC(@1);
    t->dtype = "f32";
    MulVecType *vec = ($6);
    for (auto it = vec->begin(); it != vec->end(); it++)
        t->dims.push_back(std::move(*it));
    delete vec;
    $$ = t;
  }
  ;

TensorDims
  : ConstExp {
    auto v = new MulVecType;
    v->push_back(BaseASTPtr($1));
    $$ = v;
  }
  | TensorDims ',' ConstExp {
    MulVecType *v = ($1);
    v->push_back(BaseASTPtr($3));
    $$ = v;
  }
  ;

FuncRParams
  : Exp {
    MulVecType *vec = new MulVecType;
    vec->push_back(BaseASTPtr($1));
    $$ = vec;
  }
  | FuncRParams ',' Exp {
    MulVecType *vec = ($1);
    vec->push_back(BaseASTPtr($3));
    $$ = vec;
  }
  ;

Type
  : INT  { $$ = new string("int"); }
  | VOID { $$ = new string("void"); }
  ;

Decl
  : ConstDecl {
    auto decl = new DeclAST();
    decl->loc = LOC(@1);
    decl->def = DeclAST::def_const;
    decl->decl = BaseASTPtr($1);
    $$ = decl;
  }
  | VarDecl {
    auto decl = new DeclAST();
    decl->loc = LOC(@1);
    decl->def = DeclAST::def_var;
    decl->decl = BaseASTPtr($1);
    $$ = decl;
  }
  ;

ConstDecl
  : CONST Type ConstDefs ';' {
    auto cd = new ConstDeclAST();
    cd->loc = LOC(@1);
    cd->bType = *unique_ptr<string>($2);
    MulVecType *vec = ($3);
    for (auto it = vec->begin(); it != vec->end(); it++)
      cd->constDefs.push_back(std::move(*it));
    delete vec;
    $$ = cd;
  }
  ;

ConstDefs
  : ConstDef {
    auto v = new MulVecType;
    v->push_back(BaseASTPtr($1));
    $$ = v;
  }
  | ConstDefs ',' ConstDef {
    MulVecType *v = ($1);
    v->push_back(BaseASTPtr($3));
    $$ = v;
  }
  ;

ConstDef
  : IDENT '=' ConstInitVal {
    auto d = new ConstDefAST();
    d->loc = LOC(@1);
    d->ident = *unique_ptr<string>($1);
    d->constInitVal = BaseASTPtr($3);
    $$ = d;
  }
  | IDENT ConstExpArray '=' ConstInitVal {
    auto d = new ConstDefAST();
    d->loc = LOC(@1);
    d->ident = *unique_ptr<string>($1);
    MulVecType *vec = ($2);
    for (auto it = vec->begin(); it != vec->end(); it++)
        d->constExpArray.push_back(std::move(*it));
    delete vec;
    d->constInitVal = BaseASTPtr($4);
    d->isArray = 1;
    $$ = d;
  }
  ;

ConstInitVal
  : ConstExp {
    auto v = new ConstInitValAST();
    v->loc = LOC(@1);
    v->subExp = BaseASTPtr($1);
    $$ = v;
  }
  | '{' '}' {
    auto v = new ConstInitValAST();
    v->loc = LOC(@1);
    v->isArray = 1;
    v->isEmptyInitArray = 1;
    $$ = v;
  }
  | '{' ConstInitVals '}' {
    auto v = new ConstInitValAST();
    v->loc = LOC(@1);
    v->isArray = 1;
    MulVecType *vec = ($2);
    for (auto it = vec->begin(); it != vec->end(); it++)
        v->constInitVals.push_back(std::move(*it));
    delete vec;
    $$ = v;
  }
  ;

ConstInitVals
  : ConstInitVal {
    auto v = new MulVecType;
    v->push_back(BaseASTPtr($1));
    $$ = v;
  }
  | ConstInitVals ',' ConstInitVal {
    MulVecType *v = ($1);
    v->push_back(BaseASTPtr($3));
    $$ = v;
  }
  ;

VarDecl
  : Type VarDefs ';' {
    auto vd = new VarDeclAST();
    vd->loc = LOC(@1);
    vd->bType = *unique_ptr<string>($1);
    MulVecType *vec = ($2);
    for (auto it = vec->begin(); it != vec->end(); it++)
        vd->varDefs.push_back(std::move(*it));
    delete vec;
    $$ = vd;
  }
  | TensorType IDENT ';' {
    auto vd = new VarDeclAST();
    vd->loc = LOC(@1);
    vd->bType = "tensor";
    auto v = new VarDefAST();
    v->loc = LOC(@2);
    v->ident = *unique_ptr<string>($2);
    v->tensorType = BaseASTPtr($1);
    vd->varDefs.push_back(BaseASTPtr(v));
    $$ = vd;
  }
  | TensorType IDENT '=' InitVal ';' {
    auto vd = new VarDeclAST();
    vd->loc = LOC(@1);
    vd->bType = "tensor";
    auto v = new VarDefAST();
    v->loc = LOC(@2);
    v->ident = *unique_ptr<string>($2);
    v->isInitialized = 1;
    v->tensorType = BaseASTPtr($1);
    v->initVal = BaseASTPtr($4);
    vd->varDefs.push_back(BaseASTPtr(v));
    $$ = vd;
  }
  ;

VarDefs
  : VarDef {
    auto v = new MulVecType;
    v->push_back(BaseASTPtr($1));
    $$ = v;
  }
  | VarDefs ',' VarDef {
    MulVecType *v = ($1);
    v->push_back(BaseASTPtr($3));
    $$ = v;
  }
  ;

VarDef
  : IDENT {
    auto v = new VarDefAST();
    v->loc = LOC(@1);
    v->ident = *unique_ptr<string>($1);
    $$ = v;
  }
  | IDENT '=' InitVal {
    auto v = new VarDefAST();
    v->loc = LOC(@1);
    v->isInitialized = 1;
    v->ident = *unique_ptr<string>($1);
    v->initVal = BaseASTPtr($3);
    $$ = v;
  }
  | IDENT ConstExpArray {
    auto v = new VarDefAST();
    v->loc = LOC(@1);
    v->ident = *unique_ptr<string>($1);
    MulVecType *vec = ($2);
    for (auto it = vec->begin(); it != vec->end(); it++)
        v->constExpArray.push_back(std::move(*it));
    delete vec;
    $$ = v;
  }
  | IDENT ConstExpArray '=' InitVal {
    auto v = new VarDefAST();
    v->loc = LOC(@1);
    v->isInitialized = 1;
    v->ident = *unique_ptr<string>($1);
    MulVecType *vec = ($2);
    for (auto it = vec->begin(); it != vec->end(); it++)
        v->constExpArray.push_back(std::move(*it));
    delete vec;
    v->initVal = BaseASTPtr($4);
    $$ = v;
  }
  ;

InitVal
  : Exp {
    auto v = new InitValAST();
    v->loc = LOC(@1);
    v->subExp = BaseASTPtr($1);
    $$ = v;
  }
  | '{' '}' {
    auto v = new InitValAST();
    v->loc = LOC(@1);
    v->isArray = 1;
    v->isEmptyInitArray = 1;
    $$ = v;
  }
  | '{' InitVals '}' {
    auto v = new InitValAST();
    v->loc = LOC(@1);
    MulVecType *vec = ($2);
    for (auto it = vec->begin(); it != vec->end(); it++)
        v->initVals.push_back(std::move(*it));
    delete vec;
    v->isArray = 1;
    $$ = v;
  }
  ;

InitVals
  : InitVal {
    auto v = new MulVecType;
    v->push_back(BaseASTPtr($1));
    $$ = v;
  }
  | InitVals ',' InitVal {
    MulVecType *v = ($1);
    v->push_back(BaseASTPtr($3));
    $$ = v;
  }
  ;

Block
  : '{' BlockItems '}' {
    auto b = new BlockAST();
    b->loc = LOC(@1);
    MulVecType *vec = ($2);
    for (auto it = vec->begin(); it != vec->end(); it++)
      b->blockItems.push_back(std::move(*it));
    delete vec;
    $$ = b;
  }
  | '{' '}' {
    auto b = new BlockAST();
    b->loc = LOC(@1);
    $$ = b;
  }
  ;

BlockItems
  : BlockItem {
    auto v = new MulVecType;
    v->push_back(BaseASTPtr($1));
    $$ = v;
  }
  | BlockItems BlockItem {
    MulVecType *v = ($1);
    v->push_back(BaseASTPtr($2));
    $$ = v;
  }
  ;

BlockItem
  : Decl {
    auto bi = new BlockItemAST();
    bi->loc = LOC(@1);
    bi->def = BlockItemAST::def_decl;
    bi->blockItem = BaseASTPtr($1);
    $$ = bi;
  }
  | ComplexStmt {
    auto bi = new BlockItemAST();
    bi->loc = LOC(@1);
    bi->def = BlockItemAST::def_stmt;
    bi->blockItem = BaseASTPtr($1);
    $$ = bi;
  }
  ;

ComplexStmt
  : OpenStmt   { $$ = $1; }
  | ClosedStmt { $$ = $1; }
  ;

ClosedStmt
  : Stmt {
    auto s = new ComplexStmtAST();
    s->loc = LOC(@1);
    s->def = ComplexStmtAST::def_simple;
    s->subExp = BaseASTPtr($1);
    $$ = s;
  }
  | IF '(' Exp ')' ClosedStmt ELSE ClosedStmt {
    auto s = new ComplexStmtAST();
    s->loc = LOC(@1);
    s->def = ComplexStmtAST::def_ifelse;
    s->subExp = BaseASTPtr($3);
    s->subStmt = BaseASTPtr($5);
    s->elseStmt = BaseASTPtr($7);
    $$ = s;
  }
  | WHILE '(' Exp ')' ClosedStmt {
    auto s = new ComplexStmtAST();
    s->loc = LOC(@1);
    s->def = ComplexStmtAST::def_while;
    s->subExp = BaseASTPtr($3);
    s->subStmt = BaseASTPtr($5);
    $$ = s;
  }
  ;

OpenStmt
  : IF '(' Exp ')' ComplexStmt {
    auto s = new ComplexStmtAST();
    s->loc = LOC(@1);
    s->def = ComplexStmtAST::def_openif;
    s->subExp = BaseASTPtr($3);
    s->subStmt = BaseASTPtr($5);
    $$ = s;
  }
  | IF '(' Exp ')' ClosedStmt ELSE OpenStmt {
    auto s = new ComplexStmtAST();
    s->loc = LOC(@1);
    s->def = ComplexStmtAST::def_ifelse;
    s->subExp = BaseASTPtr($3);
    s->subStmt = BaseASTPtr($5);
    s->elseStmt = BaseASTPtr($7);
    $$ = s;
  }
  | WHILE '(' Exp ')' OpenStmt {
    auto s = new ComplexStmtAST();
    s->loc = LOC(@1);
    s->def = ComplexStmtAST::def_while;
    s->subExp = BaseASTPtr($3);
    s->subStmt = BaseASTPtr($5);
    $$ = s;
  }
  ;

Stmt
  : RETURN Exp ';' {
    auto s = new StmtAST();
    s->loc = LOC(@1);
    s->def = StmtAST::def_ret;
    s->subExp = BaseASTPtr($2);
    $$ = s;
  }
  | RETURN ';' {
    auto s = new StmtAST();
    s->loc = LOC(@1);
    s->def = StmtAST::def_ret;
    $$ = s;
  }
  | LVal '=' Exp ';' {
    auto s = new StmtAST();
    s->loc = LOC(@1);
    s->def = StmtAST::def_lval;
    s->lVal = *unique_ptr<string>($1);
    s->subExp = BaseASTPtr($3);
    $$ = s;
  }
  | Exp ';' {
    auto s = new StmtAST();
    s->loc = LOC(@1);
    s->def = StmtAST::def_exp;
    s->subExp = BaseASTPtr($1);
    $$ = s;
  }
  | ';' {
    auto s = new StmtAST();
    s->loc = LOC(@1);
    s->def = StmtAST::def_exp;
    $$ = s;
  }
  | Block {
    auto s = new StmtAST();
    s->loc = LOC(@1);
    s->def = StmtAST::def_block;
    s->subExp = BaseASTPtr($1);
    $$ = s;
  }
  | BREAK ';' {
    auto s = new StmtAST();
    s->loc = LOC(@1);
    s->def = StmtAST::def_break;
    $$ = s;
  }
  | CONTINUE ';' {
    auto s = new StmtAST();
    s->loc = LOC(@1);
    s->def = StmtAST::def_continue;
    $$ = s;
  }
  | IDENT ExpArray '=' Exp ';' {
    auto s = new StmtAST();
    s->loc = LOC(@1);
    s->lVal = *unique_ptr<string>($1);
    s->def = StmtAST::def_array;
    s->subExp = BaseASTPtr($4);
    MulVecType *vec = ($2);
    for (auto it = vec->begin(); it != vec->end(); it++)
        s->expArray.push_back(std::move(*it));
    delete vec;
    $$ = s;
  }
  ;

LVal
  : IDENT {
    auto *lval = new string(*unique_ptr<string>($1));
    $$ = lval;
  }
  ;

ConstExp
  : Exp {
    auto e = new ConstExpAST();
    e->loc = LOC(@1);
    e->subExp = BaseASTPtr($1);
    $$ = e;
  }
  ;

ConstExpArray
  : '[' ConstExp ']' {
    auto v = new MulVecType;
    v->push_back(BaseASTPtr($2));
    $$ = v;
  }
  | ConstExpArray '[' ConstExp ']' {
    MulVecType *v = ($1);
    v->push_back(BaseASTPtr($3));
    $$ = v;
  }
  ;

Exp
  : LOrExp {
    auto e = new ExpAST();
    e->loc = LOC(@1);
    e->subExp = BaseASTPtr($1);
    $$ = e;
  }
  ;

ExpArray
  : '[' Exp ']' {
    auto v = new MulVecType;
    v->push_back(BaseASTPtr($2));
    $$ = v;
  }
  | ExpArray '[' Exp ']' {
    MulVecType *v = ($1);
    v->push_back(BaseASTPtr($3));
    $$ = v;
  }
  ;

PrimaryExp
  : '(' Exp ')' {
    auto p = new PrimaryExpAST();
    p->loc = LOC(@1);
    p->def = PrimaryExpAST::def_bracketexp;
    p->subExp = BaseASTPtr($2);
    $$ = p;
  }
  | LVal {
    auto p = new PrimaryExpAST();
    p->loc = LOC(@1);
    p->def = PrimaryExpAST::def_lval;
    p->lVal = *unique_ptr<string>($1);
    $$ = p;
  }
  | Number {
    auto p = new PrimaryExpAST();
    p->loc = LOC(@1);
    p->def = PrimaryExpAST::def_number;
    p->number = ($1);
    $$ = p;
  }
  | IDENT ExpArray {
    auto p = new PrimaryExpAST();
    p->loc = LOC(@1);
    p->def = PrimaryExpAST::def_array;
    p->arrayIdent = *unique_ptr<string>($1);
    MulVecType *vec = ($2);
    for (auto it = vec->begin(); it != vec->end(); it++)
        p->expArray.push_back(std::move(*it));
    delete vec;
    $$ = p;
  }
  ;

Number
  : INT_CONST { $$ = ($1); }
  ;

UnaryExp
  : PrimaryExp {
    auto u = new UnaryExpAST();
    u->loc = LOC(@1);
    u->def = UnaryExpAST::def_primaryexp;
    u->subExp = BaseASTPtr($1);
    $$ = u;
  }
  | ADDOP UnaryExp {
    auto u = new UnaryExpAST();
    u->loc = LOC(@1);
    u->def = UnaryExpAST::def_unaryexp;
    u->op = *unique_ptr<string>($1);
    u->subExp = BaseASTPtr($2);
    $$ = u;
  }
  | UNARYOP UnaryExp {
    auto u = new UnaryExpAST();
    u->loc = LOC(@1);
    u->def = UnaryExpAST::def_unaryexp;
    u->op = *unique_ptr<string>($1);
    u->subExp = BaseASTPtr($2);
    $$ = u;
  }
  | IDENT '(' ')' {
    auto u = new UnaryExpAST();
    u->loc = LOC(@1);
    u->def = UnaryExpAST::def_func;
    u->ident = *unique_ptr<string>($1);
    $$ = u;
  }
  | IDENT '(' FuncRParams ')' {
    auto u = new UnaryExpAST();
    u->loc = LOC(@1);
    u->def = UnaryExpAST::def_func;
    u->ident = *unique_ptr<string>($1);
    MulVecType *vec = ($3);
    for (auto it = vec->begin(); it != vec->end(); it++)
        u->funcRParams.push_back(std::move(*it));
    delete vec;
    $$ = u;
  }
  | AT IDENT '(' ')' {
    auto u = new UnaryExpAST();
    u->loc = LOC(@1);
    u->def = UnaryExpAST::def_builtin;
    u->ident = *unique_ptr<string>($2);
    $$ = u;
  }
  | AT IDENT '(' FuncRParams ')' {
    auto u = new UnaryExpAST();
    u->loc = LOC(@1);
    u->def = UnaryExpAST::def_builtin;
    u->ident = *unique_ptr<string>($2);
    MulVecType *vec = ($4);
    for (auto it = vec->begin(); it != vec->end(); it++)
        u->funcRParams.push_back(std::move(*it));
    delete vec;
    $$ = u;
  }
  ;

MulExp
  : UnaryExp {
    auto e = new MulExpAST();
    e->loc = LOC(@1);
    e->subExp = BaseASTPtr($1);
    $$ = e;
  }
  | MulExp MULOP UnaryExp {
    auto e = new MulExpAST();
    e->loc = LOC(@1);
    e->mulExp = BaseASTPtr($1);
    e->op = *unique_ptr<string>($2);
    e->subExp = BaseASTPtr($3);
    $$ = e;
  }
  ;

AddExp
  : MulExp {
    auto e = new AddExpAST();
    e->loc = LOC(@1);
    e->subExp = BaseASTPtr($1);
    $$ = e;
  }
  | AddExp ADDOP MulExp {
    auto e = new AddExpAST();
    e->loc = LOC(@1);
    e->addExp = BaseASTPtr($1);
    e->op = *unique_ptr<string>($2);
    e->subExp = BaseASTPtr($3);
    $$ = e;
  }
  ;

RelExp
  : AddExp {
    auto e = new RelExpAST();
    e->loc = LOC(@1);
    e->subExp = BaseASTPtr($1);
    $$ = e;
  }
  | RelExp RELOP AddExp {
    // <= or >= (two-char relops; lexer packs the literal string into yylval).
    auto e = new RelExpAST();
    e->loc = LOC(@1);
    e->relExp = BaseASTPtr($1);
    e->op = *unique_ptr<string>($2);
    e->subExp = BaseASTPtr($3);
    $$ = e;
  }
  | RelExp '<' AddExp {
    // Single '<' comes through as a character token (see sysy.l note).
    auto e = new RelExpAST();
    e->loc = LOC(@1);
    e->relExp = BaseASTPtr($1);
    e->op = "<";
    e->subExp = BaseASTPtr($3);
    $$ = e;
  }
  | RelExp '>' AddExp {
    auto e = new RelExpAST();
    e->loc = LOC(@1);
    e->relExp = BaseASTPtr($1);
    e->op = ">";
    e->subExp = BaseASTPtr($3);
    $$ = e;
  }
  ;

EqExp
  : RelExp {
    auto e = new EqExpAST();
    e->loc = LOC(@1);
    e->subExp = BaseASTPtr($1);
    $$ = e;
  }
  | EqExp EQOP RelExp {
    auto e = new EqExpAST();
    e->loc = LOC(@1);
    e->eqExp = BaseASTPtr($1);
    e->op = *unique_ptr<string>($2);
    e->subExp = BaseASTPtr($3);
    $$ = e;
  }
  ;

LAndExp
  : EqExp {
    auto e = new LAndExpAST();
    e->loc = LOC(@1);
    e->subExp = BaseASTPtr($1);
    $$ = e;
  }
  | LAndExp LANDOP EqExp {
    auto e = new LAndExpAST();
    e->loc = LOC(@1);
    e->lAndExp = BaseASTPtr($1);
    e->op = *unique_ptr<string>($2);
    e->subExp = BaseASTPtr($3);
    $$ = e;
  }
  ;

LOrExp
  : LAndExp {
    auto e = new LOrExpAST();
    e->loc = LOC(@1);
    e->subExp = BaseASTPtr($1);
    $$ = e;
  }
  | LOrExp LOROP LAndExp {
    auto e = new LOrExpAST();
    e->loc = LOC(@1);
    e->lOrExp = BaseASTPtr($1);
    e->op = *unique_ptr<string>($2);
    e->subExp = BaseASTPtr($3);
    $$ = e;
  }
  ;

%%

void yyerror(BaseASTPtr & /*ast*/, DiagnosticEngine &diag,
             const std::string &filename, const char *s) {
    SourceLocation loc{filename, yylloc.first_line, yylloc.first_column};
    diag.error(loc, std::string("parse error: ") + (s ? s : "syntax error"));
}
