#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../frontend/location.h"
#include "types.h"

namespace tsy::hir {

enum class OpKind {
    Param,      // pseudo-op for function parameters (not added to ops list).
    MatMul,
    Add,
    Softmax,
    RMSNorm,
    View,       // W1 syntax not yet exposed; reserved.
    Permute,    // ditto.
    FuncCall,   // non-builtin function call.
    Return,
    Unknown,    // fallback for unhandled AST shapes during lowering.
};

const char* toString(OpKind);

// Turns a TensorSysY builtin name ("matmul", "add", ...) into a concrete
// OpKind, or OpKind::Unknown if the name isn't recognised. Useful both for
// lowering and for verifier diagnostics.
OpKind builtinKindFromName(const std::string& name);

struct Op;

struct Value {
    std::string name;       // SSA-style, e.g. "%A", "%0".
    TensorType type;
    Op* defining_op = nullptr;  // null for function parameters.
};
using ValuePtr = std::shared_ptr<Value>;

struct Op {
    OpKind kind = OpKind::Unknown;
    std::string builtin_name;   // populated for Unknown / FuncCall cases so
                                 // the printer can render the original name.
    tsy::SourceLocation loc;
    std::vector<ValuePtr> operands;
    std::vector<ValuePtr> results;
};

struct Function {
    std::string name;
    std::string return_type;           // "int" / "void"
    std::vector<ValuePtr> params;      // includes non-tensor params; tensor
                                       // ones get their TensorType set.
    std::vector<std::unique_ptr<Op>> ops;
    tsy::SourceLocation loc;
};

struct Module {
    std::vector<std::unique_ptr<Function>> funcs;
};

}  // namespace tsy::hir
