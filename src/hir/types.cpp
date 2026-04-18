#include "types.h"

namespace tsy::hir {

const char* toString(DType d) {
    switch (d) {
        case DType::F32: return "f32";
    }
    return "?";
}

std::string Dim::format() const {
    if (resolved) {
        // Show "M=8" when we know the symbol AND its value, but hide the
        // symbol for pure numeric literals (they'd look like "8=8").
        if (!symbol.empty() && symbol != std::to_string(*resolved)) {
            return symbol + "=" + std::to_string(*resolved);
        }
        return std::to_string(*resolved);
    }
    return symbol.empty() ? std::string("?") : symbol;
}

bool Shape::allResolved() const {
    for (const auto& d : dims)
        if (!d.resolved) return false;
    return true;
}

std::vector<int64_t> Shape::resolvedOrZero() const {
    std::vector<int64_t> out;
    out.reserve(dims.size());
    for (const auto& d : dims) out.push_back(d.resolved.value_or(0));
    return out;
}

std::string formatType(const TensorType& t) {
    std::string s = "tensor<";
    s += toString(t.dtype);
    s += ">[";
    for (size_t i = 0; i < t.shape.dims.size(); ++i) {
        if (i) s += ", ";
        s += t.shape.dims[i].format();
    }
    s += "]";
    return s;
}

}  // namespace tsy::hir
