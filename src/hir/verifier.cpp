#include "verifier.h"

#include <sstream>

using namespace tsy;

namespace tsy::hir {

namespace {

bool isTensor(const Value& v) { return !v.type.shape.empty(); }

std::string shapeAsString(const Shape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) {
        if (i) out += ", ";
        out += s.dims[i].format();
    }
    out += "]";
    return out;
}

bool dimsEqual(const Dim& a, const Dim& b) {
    // Prefer comparing resolved values (strict). Symbol equality is only
    // used as a fallback when resolution failed on both sides, which the
    // caller already flags via the "unresolved dim" check.
    if (a.resolved && b.resolved) return *a.resolved == *b.resolved;
    if (a.resolved || b.resolved) return false;
    return !a.symbol.empty() && a.symbol == b.symbol;
}

bool shapesEqual(const Shape& a, const Shape& b) {
    if (a.rank() != b.rank()) return false;
    for (size_t i = 0; i < a.rank(); ++i) {
        if (!dimsEqual(a.dims[i], b.dims[i])) return false;
    }
    return true;
}

bool requireResolved(const Value& v, DiagnosticEngine& diag, SourceLocation loc,
                     const char* role) {
    if (!isTensor(v)) return true;  // scalars don't carry shape in W3.
    if (v.type.shape.allResolved()) return true;
    std::ostringstream oss;
    oss << role << " tensor '" << v.name << "' has unresolved shape "
        << shapeAsString(v.type.shape) << "; every dim must be a compile-time "
                                          "constant (add `const int` decls "
                                          "before the tensor type).";
    diag.error(loc, oss.str());
    return false;
}

bool requireTensor(const Value& v, DiagnosticEngine& diag, SourceLocation loc,
                   const char* role, const char* opName) {
    if (isTensor(v)) return true;
    std::ostringstream oss;
    oss << opName << ": " << role << " operand '" << v.name
        << "' is not a tensor; scalar operands are not supported.";
    diag.error(loc, oss.str());
    return false;
}

// --- op-specific checks ----------------------------------------------------

void verifyMatMul(const Op& op, DiagnosticEngine& diag) {
    if (op.operands.size() != 2) {
        std::ostringstream oss;
        oss << "matmul expects 2 operands, got " << op.operands.size();
        diag.error(op.loc, oss.str());
        return;
    }
    const auto& a = *op.operands[0];
    const auto& b = *op.operands[1];
    if (!requireTensor(a, diag, op.loc, "lhs", "matmul")) return;
    if (!requireTensor(b, diag, op.loc, "rhs", "matmul")) return;
    if (!requireResolved(a, diag, op.loc, "matmul lhs")) return;
    if (!requireResolved(b, diag, op.loc, "matmul rhs")) return;

    if (a.type.shape.rank() < 2 || b.type.shape.rank() < 2) {
        std::ostringstream oss;
        oss << "matmul requires rank >= 2 on both sides, got lhs rank "
            << a.type.shape.rank() << ", rhs rank " << b.type.shape.rank();
        diag.error(op.loc, oss.str());
        return;
    }

    const auto& aDims = a.type.shape.dims;
    const auto& bDims = b.type.shape.dims;
    const Dim& aInner = aDims[aDims.size() - 1];
    const Dim& bInner = bDims[bDims.size() - 2];
    if (!dimsEqual(aInner, bInner)) {
        std::ostringstream oss;
        oss << "matmul inner dim mismatch: lhs " << shapeAsString(a.type.shape)
            << " (inner=" << aInner.format() << ") vs rhs "
            << shapeAsString(b.type.shape) << " (inner=" << bInner.format()
            << ").";
        diag.error(op.loc, oss.str());
        return;
    }

    if (op.results.size() != 1) {
        diag.error(op.loc, "matmul must produce exactly 1 result");
        return;
    }
    const auto& r = *op.results[0];
    if (!requireTensor(r, diag, op.loc, "result", "matmul")) return;
    if (!requireResolved(r, diag, op.loc, "matmul result")) return;

    // Expected result shape: [...lhs[:-1], rhs[-1]].
    Shape expected;
    for (size_t i = 0; i + 1 < aDims.size(); ++i) expected.dims.push_back(aDims[i]);
    expected.dims.push_back(bDims[bDims.size() - 1]);
    if (!shapesEqual(r.type.shape, expected)) {
        std::ostringstream oss;
        oss << "matmul result shape " << shapeAsString(r.type.shape)
            << " disagrees with expected " << shapeAsString(expected)
            << " for lhs=" << shapeAsString(a.type.shape)
            << ", rhs=" << shapeAsString(b.type.shape) << ".";
        diag.error(op.loc, oss.str());
    }
}

void verifyAdd(const Op& op, DiagnosticEngine& diag) {
    if (op.operands.size() != 2) {
        std::ostringstream oss;
        oss << "add expects 2 operands, got " << op.operands.size();
        diag.error(op.loc, oss.str());
        return;
    }
    const auto& a = *op.operands[0];
    const auto& b = *op.operands[1];
    if (!requireTensor(a, diag, op.loc, "lhs", "add")) return;
    if (!requireTensor(b, diag, op.loc, "rhs", "add")) return;
    if (!requireResolved(a, diag, op.loc, "add lhs")) return;
    if (!requireResolved(b, diag, op.loc, "add rhs")) return;
    if (a.type.dtype != b.type.dtype) {
        diag.error(op.loc, "add: dtype mismatch");
        return;
    }
    if (!shapesEqual(a.type.shape, b.type.shape)) {
        std::ostringstream oss;
        oss << "add: elementwise requires identical shapes; got "
            << shapeAsString(a.type.shape) << " and "
            << shapeAsString(b.type.shape)
            << ". Broadcasting is not supported in W3.";
        diag.error(op.loc, oss.str());
        return;
    }
    if (op.results.size() != 1) {
        diag.error(op.loc, "add must produce exactly 1 result");
        return;
    }
    if (!shapesEqual(op.results[0]->type.shape, a.type.shape)) {
        diag.error(op.loc, "add result shape disagrees with operand shape.");
    }
}

void verifyUnary(const Op& op, DiagnosticEngine& diag, const char* name) {
    if (op.operands.size() != 1) {
        std::ostringstream oss;
        oss << name << " expects 1 operand, got " << op.operands.size();
        diag.error(op.loc, oss.str());
        return;
    }
    const auto& a = *op.operands[0];
    if (!requireTensor(a, diag, op.loc, "input", name)) return;
    if (!requireResolved(a, diag, op.loc, name)) return;
    if (op.results.size() != 1) {
        std::ostringstream oss;
        oss << name << " must produce exactly 1 result";
        diag.error(op.loc, oss.str());
        return;
    }
    if (!shapesEqual(op.results[0]->type.shape, a.type.shape)) {
        std::ostringstream oss;
        oss << name << " result shape disagrees with input shape.";
        diag.error(op.loc, oss.str());
    }
}

void verifyUnknown(const Op& op, DiagnosticEngine& diag) {
    // Only surface a diagnostic when the unknown op carries a builtin-like
    // name (i.e. the user wrote @foo and we didn't recognise it). Reasons
    // that the lowerer tags (e.g. "complex-init") are internal and fire
    // before a real builtin name is visible, so we report those differently.
    static const char* internalTags[] = {
        "complex-init", "call-init", "non-builtin-init", "tensor-array-init",
    };
    for (const char* tag : internalTags) {
        if (op.builtin_name == tag) return;  // already had its moment.
    }
    if (op.builtin_name.empty()) return;
    diag.error(op.loc, "unknown builtin '@" + op.builtin_name + "'");
}

}  // namespace

void verifyModule(const Module& m, DiagnosticEngine& diag) {
    for (const auto& f : m.funcs) {
        for (const auto& opPtr : f->ops) {
            const auto& op = *opPtr;
            switch (op.kind) {
                case OpKind::MatMul:  verifyMatMul(op, diag); break;
                case OpKind::Add:     verifyAdd(op, diag); break;
                case OpKind::Softmax: verifyUnary(op, diag, "softmax"); break;
                case OpKind::RMSNorm: verifyUnary(op, diag, "rmsnorm"); break;
                case OpKind::Unknown: verifyUnknown(op, diag); break;
                default:
                    break;
            }
        }
    }
}

}  // namespace tsy::hir
