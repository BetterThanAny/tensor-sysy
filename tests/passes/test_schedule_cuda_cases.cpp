// W9 ScheduleCudaPass unit test — constructs LIR modules programmatically
// (no parser involved) and asserts runScheduleCuda writes the expected
// variant for each shape class.

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../src/frontend/diagnostics.h"
#include "../../src/lir/ir.h"
#include "../../src/passes/pass_manager.h"

using tsy::DiagnosticEngine;
using tsy::lir::Buffer;
using tsy::lir::Function;
using tsy::lir::Module;
using tsy::lir::Stmt;
using tsy::lir::StmtKind;

namespace {

int g_failures = 0;

void fail(const std::string& label, const std::string& why) {
    std::cerr << "FAIL[" << label << "]: " << why << "\n";
    ++g_failures;
}

// Build a minimal LIR module with one function, three buffers (A[M,K],
// B[K,N], C[M,N]), and a single matmul Call.
std::unique_ptr<Module> makeMatmulModule(int64_t M, int64_t K, int64_t N) {
    auto mod = std::make_unique<Module>();
    auto f = std::make_unique<Function>();
    f->name = "matmul_case";
    f->return_type = "void";

    Buffer bufA; bufA.id = 0; bufA.name = "A"; bufA.dims = {M, K};
    Buffer bufB; bufB.id = 1; bufB.name = "B"; bufB.dims = {K, N};
    Buffer bufC; bufC.id = 2; bufC.name = "C"; bufC.dims = {M, N};
    f->buffers = {bufA, bufB, bufC};
    f->params = {0, 1};

    Stmt s;
    s.kind = StmtKind::Call;
    s.primitive = "matmul";
    s.operand_bufs = {0, 1};
    s.result_buf = 2;
    f->body.push_back(std::move(s));

    Stmt ret;
    ret.kind = StmtKind::Return;
    f->body.push_back(std::move(ret));

    mod->funcs.push_back(std::move(f));
    return mod;
}

void checkVariant(const std::string& label,
                   int64_t M, int64_t K, int64_t N,
                   const std::string& expected) {
    auto mod = makeMatmulModule(M, K, N);
    DiagnosticEngine diag;
    tsy::passes::runScheduleCuda(*mod, diag);
    if (diag.hasErrors()) {
        fail(label, "diagnostic engine reported errors");
        return;
    }
    const auto& s = mod->funcs[0]->body[0];
    auto it = s.attrs.find("variant");
    if (it == s.attrs.end()) {
        fail(label, "no variant attr written");
        return;
    }
    if (it->second != expected) {
        fail(label, "got variant='" + it->second +
                     "', expected '" + expected + "'");
    }
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    // Tiny: M*N < 1024 → naive
    checkVariant("tiny-4x4x4",   4,   4,  4, "naive");
    checkVariant("row-1x32x8",   1,  32,  8, "naive");

    // Sweet spot: aligned + large_enough + M*N <= 256^2 → tiled
    checkVariant("sweet-128x128x128",  128, 128, 128, "tiled");
    checkVariant("sweet-256x128x256",  256, 128, 256, "tiled");

    // Too big: aligned but M*N > 256^2 → cublas
    checkVariant("large-512x512x512", 512, 512, 512, "cublas");

    // Odd shape: M*N >= 1024 but alignment fails (M not multiple of 128) → cublas
    checkVariant("odd-32x13x32", 32, 13, 32, "cublas");

    // Aligned but K<128 (not large_enough) → cublas
    checkVariant("small-K-128x64x128", 128, 64, 128, "cublas");

    if (g_failures == 0) {
        std::cout << "pass_schedule_cuda_cases: ALL PASS\n";
        return 0;
    }
    std::cerr << "pass_schedule_cuda_cases: " << g_failures
              << " failure(s)\n";
    return 1;
}
