#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../frontend/location.h"
#include "../hir/types.h"  // DType

namespace tsy::lir {

// Buffer is a materialised tensor storage slot owned by a Function.
// The LIR keeps only the shape metadata here; concrete bytes live in the
// interpreter's runtime arena, keyed by the buffer's index.
struct Buffer {
    int id = 0;                    // index into Function::buffers.
    std::string name;              // source-level name (e.g. "A", "C").
    std::vector<int64_t> dims;     // fully resolved, static.
    hir::DType dtype = hir::DType::F32;

    int64_t numElements() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
};

enum class StmtKind {
    Call,       // primitive_name(operands...) -> result_buf
    Return,     // end of function; no value tracking in W4.
};

// Thin "primitive-call" IR. Each Call references buffer indices instead of
// pointers so the Function is trivially copyable/serialisable. Loop-level
// machinery (indexing, tiled schedules) lives inside the interpreter's
// primitive implementations for W4; LIR will grow loop structs in later
// weeks when codegen needs them.
struct Stmt {
    StmtKind kind = StmtKind::Call;
    std::string primitive;         // "matmul" / "add" / "softmax" / ...
    std::vector<int> operand_bufs; // indices into Function::buffers.
    int result_buf = -1;           // index into Function::buffers; -1 = none.
    tsy::SourceLocation loc;

    // Generic attribute map. Passes write into this (e.g. ScheduleCudaPass
    // sets "variant" for matmul calls). The LIR printer emits these as a
    // trailing `{k="v", ...}` group, sorted by key for golden stability.
    // Default-empty preserves pre-W9 behaviour (printouts / codegen are
    // byte-identical to W8 when no pass writes into attrs).
    std::unordered_map<std::string, std::string> attrs;
};

struct Function {
    std::string name;
    std::string return_type;       // "int" / "void"
    std::vector<Buffer> buffers;   // tensor buffers; scalar params are omitted.
    std::vector<int> params;       // indices into `buffers`; order preserved.
    std::vector<Stmt> body;
    tsy::SourceLocation loc;
};

struct Module {
    std::vector<std::unique_ptr<Function>> funcs;
};

}  // namespace tsy::lir
