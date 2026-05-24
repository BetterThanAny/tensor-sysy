%option noyywrap
%option nounput
%option noinput
%option yylineno

%{

#include <cstdlib>
#include <string>

#include "ast.h"
#include "sysy.tab.hpp"

using namespace std;

static int yycolumn = 1;

static void updateLocation(const char* text, int len) {
    yylloc.first_line = yylineno;
    yylloc.first_column = yycolumn;
    for (int i = 0; i < len; ++i) {
        if (text[i] == '\n') yycolumn = 1;
        else ++yycolumn;
    }
    yylloc.last_line = yylineno;
    yylloc.last_column = yycolumn;
}

#define YY_USER_ACTION updateLocation(yytext, yyleng);

%}

/* White space & comments */
WhiteSpace    [ \t\n\r]*
LineComment   "//".*
BlockComment  "/*"([^*]*|(\*+[^/]))*"*/"

/* Identifier */
Identifier    [a-zA-Z_][a-zA-Z0-9_]*

/* Integer */
Decimal       [1-9][0-9]*
Octal         0[0-7]*
Hexadecimal   0[xX][0-9a-fA-F]+

/* Operator */
UnaryOperator \!
MulOperator   \*|\/|\%
AddOperator   \+|\-
/* Two-char relops only; single '<' and '>' are returned as characters so
 * `tensor<f32>` can use them as delimiters. */
RelOperator   <=|>=
EqOperator    ==|\!=
LAndOperator  &&
LOrOperator   \|\|


%%

{WhiteSpace}    { /* skip */ }
{LineComment}   { /* skip */ }
{BlockComment}  { /* skip */ }

"int"           { return INT; }
"return"        { return RETURN; }
"const"         { return CONST; }
"void"          { return VOID; }
"if"            { return IF; }
"else"          { return ELSE; }
"while"         { return WHILE; }
"break"         { return BREAK; }
"continue"      { return CONTINUE; }

"tensor"        { return TENSOR; }
"f32"           { return F32; }
"@"             { return AT; }

{Identifier}    { yylval.str_val = new string(yytext); return IDENT; }

{Decimal}       { yylval.int_val = strtoll(yytext, nullptr, 0); return INT_CONST; }
{Octal}         { yylval.int_val = strtoll(yytext, nullptr, 0); return INT_CONST; }
{Hexadecimal}   { yylval.int_val = strtoll(yytext, nullptr, 0); return INT_CONST; }

{UnaryOperator} { yylval.str_val = new string(yytext); return UNARYOP; }
{MulOperator}   { yylval.str_val = new string(yytext); return MULOP; }
{AddOperator}   { yylval.str_val = new string(yytext); return ADDOP; }
{RelOperator}   { yylval.str_val = new string(yytext); return RELOP; }
{EqOperator}    { yylval.str_val = new string(yytext); return EQOP; }
{LAndOperator}  { yylval.str_val = new string(yytext); return LANDOP; }
{LOrOperator}   { yylval.str_val = new string(yytext); return LOROP; }

.               { return yytext[0]; }

%%
