#include "printer.h"

#include <sstream>

namespace tsy::lir {

namespace {

void spaces(std::ostream& os, int n) {
    for (int i = 0; i < n; ++i) os.put(' ');
}

std::string shape(const Buffer& b) {
    std::string out = "[";
    for (size_t i = 0; i < b.dims.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(b.dims[i]);
    }
    out += "]";
    return out;
}

}  // namespace

void printFunction(std::ostream& os, const Function& f, int indent) {
    spaces(os, indent);
    os << "func @" << f.name << "(";
    for (size_t i = 0; i < f.params.size(); ++i) {
        if (i) os << ", ";
        const auto& b = f.buffers[f.params[i]];
        os << "%" << b.name << ": tensor<" << toString(b.dtype) << ">" << shape(b);
    }
    os << ") -> " << (f.return_type.empty() ? std::string("void") : f.return_type);
    os << " {\n";

    // Buffer table first so call-sites stay readable.
    for (const auto& b : f.buffers) {
        bool is_param = false;
        for (int p : f.params) if (p == b.id) { is_param = true; break; }
        spaces(os, indent + 2);
        os << "buf #" << b.id << " %" << b.name << " : tensor<"
           << toString(b.dtype) << ">" << shape(b)
           << (is_param ? "  (param)" : "") << "\n";
    }

    for (const auto& s : f.body) {
        spaces(os, indent + 2);
        if (s.kind == StmtKind::Return) {
            os << "return\n";
            continue;
        }
        if (s.result_buf >= 0) {
            os << "%" << f.buffers[s.result_buf].name << " = ";
        }
        os << "call " << s.primitive;
        for (size_t i = 0; i < s.operand_bufs.size(); ++i) {
            os << (i == 0 ? " " : ", ");
            os << "%" << f.buffers[s.operand_bufs[i]].name;
        }
        os << "\n";
    }

    spaces(os, indent);
    os << "}\n";
}

void printModule(std::ostream& os, const Module& m) {
    os << "lir.module {\n";
    for (const auto& f : m.funcs) printFunction(os, *f);
    os << "}\n";
}

}  // namespace tsy::lir
