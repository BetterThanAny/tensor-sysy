#include "printer.h"

#include <sstream>

namespace tsy::hir {

namespace {

void spaces(std::ostream& os, int n) {
    for (int i = 0; i < n; ++i) os.put(' ');
}

bool isTensor(const Value& v) { return !v.type.shape.empty(); }

std::string formatValueWithType(const Value& v) {
    if (isTensor(v)) return v.name + ": " + formatType(v.type);
    return v.name;  // scalars don't carry a tensor type in W2.
}

}  // namespace

void printOp(std::ostream& os, const Op& op, int indent) {
    spaces(os, indent);

    // Results, if any, go on the LHS of an '=': `%r0, %r1 = matmul ...`.
    for (size_t i = 0; i < op.results.size(); ++i) {
        if (i) os << ", ";
        os << op.results[i]->name;
    }
    if (!op.results.empty()) os << " = ";

    // Op name. Unknown ops print with the AST name preserved so the HIR dump
    // is diagnosable even when lowering doesn't recognise a builtin.
    if (op.kind == OpKind::Unknown && !op.builtin_name.empty()) {
        os << "unknown[" << op.builtin_name << "]";
    } else {
        os << toString(op.kind);
        if (op.kind == OpKind::FuncCall && !op.builtin_name.empty()) {
            os << " @" << op.builtin_name;
        }
    }

    // Operands.
    if (!op.operands.empty()) os << " ";
    for (size_t i = 0; i < op.operands.size(); ++i) {
        if (i) os << ", ";
        os << op.operands[i]->name;
    }

    // Result type annotation. Bias toward the first result's type since the
    // W2 ops we emit all produce a single tensor.
    if (!op.results.empty() && isTensor(*op.results.front())) {
        os << " : " << formatType(op.results.front()->type);
    }
    os << "\n";
}

void printFunction(std::ostream& os, const Function& f, int indent) {
    spaces(os, indent);
    os << "func @" << f.name << "(";
    for (size_t i = 0; i < f.params.size(); ++i) {
        if (i) os << ", ";
        os << formatValueWithType(*f.params[i]);
    }
    os << ") -> " << (f.return_type.empty() ? std::string("void") : f.return_type);
    os << " {\n";
    for (const auto& op : f.ops) printOp(os, *op, indent + 2);
    spaces(os, indent);
    os << "}\n";
}

void printModule(std::ostream& os, const Module& m) {
    os << "module {\n";
    for (const auto& f : m.funcs) printFunction(os, *f);
    os << "}\n";
}

}  // namespace tsy::hir
