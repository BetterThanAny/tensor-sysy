#include "lowering.h"

#include <sstream>
#include <unordered_map>

using namespace tsy;
using namespace tsy::hir;

namespace tsy::lir {

namespace {

bool isTensor(const hir::Value& v) { return !v.type.shape.empty(); }

std::vector<int64_t> resolvedDims(const hir::Shape& s) {
    std::vector<int64_t> out;
    out.reserve(s.dims.size());
    for (const auto& d : s.dims) out.push_back(d.resolved.value_or(0));
    return out;
}

struct Lowerer {
    DiagnosticEngine& diag;
    std::unique_ptr<Module> out = std::make_unique<Module>();

    explicit Lowerer(DiagnosticEngine& d) : diag(d) {}

    void run(const hir::Module& m) {
        for (const auto& f : m.funcs) lowerFunction(*f);
    }

    // Map an HIR Value pointer to its freshly minted LIR buffer id.
    using ValueMap = std::unordered_map<const hir::Value*, int>;

    int materialiseBuffer(Function& f, ValueMap& vm, const hir::Value& v) {
        auto it = vm.find(&v);
        if (it != vm.end()) return it->second;
        Buffer b;
        b.id = static_cast<int>(f.buffers.size());
        // Strip the SSA `%` prefix the HIR printer uses to keep LIR names
        // source-faithful (e.g. "A", "C", "sum").
        b.name = !v.name.empty() && v.name.front() == '%' ? v.name.substr(1) : v.name;
        b.dims = resolvedDims(v.type.shape);
        b.dtype = v.type.dtype;
        f.buffers.push_back(std::move(b));
        vm[&v] = f.buffers.back().id;
        return f.buffers.back().id;
    }

    void lowerFunction(const hir::Function& fn) {
        auto lf = std::make_unique<Function>();
        lf->name = fn.name;
        lf->return_type = fn.return_type;
        lf->loc = fn.loc;
        ValueMap vm;

        // Tensor params first; scalar params get no buffer in W4.
        for (const auto& p : fn.params) {
            if (!isTensor(*p)) continue;
            int id = materialiseBuffer(*lf, vm, *p);
            lf->params.push_back(id);
        }

        for (const auto& opPtr : fn.ops) {
            const auto& op = *opPtr;
            if (op.kind == hir::OpKind::Return) {
                Stmt s;
                s.kind = StmtKind::Return;
                s.loc = op.loc;
                lf->body.push_back(std::move(s));
                continue;
            }
            // W4 only knows how to run the four recognised builtins.
            switch (op.kind) {
                case hir::OpKind::MatMul:
                case hir::OpKind::Add:
                case hir::OpKind::Softmax:
                case hir::OpKind::RMSNorm:
                case hir::OpKind::Transpose:
                case hir::OpKind::ReLU:
                    lowerBuiltinCall(*lf, vm, op);
                    break;
                default: {
                    std::ostringstream oss;
                    oss << "LIR lowering skipped " << toString(op.kind)
                        << " op (not yet supported in W4)";
                    diag.error(op.loc, oss.str());
                    break;
                }
            }
        }

        out->funcs.push_back(std::move(lf));
    }

    void lowerBuiltinCall(Function& f, ValueMap& vm, const hir::Op& op) {
        Stmt s;
        s.kind = StmtKind::Call;
        s.primitive = toString(op.kind);
        s.loc = op.loc;
        for (const auto& v : op.operands) {
            s.operand_bufs.push_back(materialiseBuffer(f, vm, *v));
        }
        if (!op.results.empty()) {
            s.result_buf = materialiseBuffer(f, vm, *op.results.front());
        }
        f.body.push_back(std::move(s));
    }
};

}  // namespace

std::unique_ptr<Module> lowerHirToLir(const hir::Module& m, DiagnosticEngine& diag) {
    Lowerer l(diag);
    l.run(m);
    return std::move(l.out);
}

}  // namespace tsy::lir
