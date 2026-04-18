#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tsy::hir {

enum class DType { F32 };

const char* toString(DType);

// W2 keeps dims symbolic (the original text from the AST, e.g. "M", "4").
// W3 adds `resolved` — the integer value a const evaluator found for that
// symbol. Printer prefers `resolved` when present; the symbol stays around
// for diagnostics ("shape [M, N] resolved to [8, 4]").
struct Dim {
    std::string symbol;
    std::optional<int64_t> resolved;

    std::string format() const;
};

struct Shape {
    std::vector<Dim> dims;

    size_t rank() const { return dims.size(); }
    bool empty() const { return dims.empty(); }
    bool allResolved() const;
    std::vector<int64_t> resolvedOrZero() const;
};

struct TensorType {
    DType dtype = DType::F32;
    Shape shape;
};

std::string formatType(const TensorType&);

}  // namespace tsy::hir
