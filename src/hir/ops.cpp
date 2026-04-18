#include "ops.h"

namespace tsy::hir {

const char* toString(OpKind k) {
    switch (k) {
        case OpKind::Param:    return "param";
        case OpKind::MatMul:   return "matmul";
        case OpKind::Add:      return "add";
        case OpKind::Softmax:  return "softmax";
        case OpKind::RMSNorm:  return "rmsnorm";
        case OpKind::View:     return "view";
        case OpKind::Permute:  return "permute";
        case OpKind::FuncCall: return "call";
        case OpKind::Return:   return "return";
        case OpKind::Unknown:  return "unknown";
    }
    return "?";
}

OpKind builtinKindFromName(const std::string& name) {
    if (name == "matmul")  return OpKind::MatMul;
    if (name == "add")     return OpKind::Add;
    if (name == "softmax") return OpKind::Softmax;
    if (name == "rmsnorm") return OpKind::RMSNorm;
    if (name == "view")    return OpKind::View;
    if (name == "permute") return OpKind::Permute;
    return OpKind::Unknown;
}

}  // namespace tsy::hir
