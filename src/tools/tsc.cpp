#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../codegen/cpp.h"
#include "../codegen/cuda.h"
#include "../hir/lowering.h"
#include "../hir/printer.h"
#include "../hir/verifier.h"
#include "../lir/interpreter.h"
#include "../lir/lowering.h"
#include "../lir/printer.h"
#include "../passes/pass_manager.h"
#include "parser_driver.h"

#if TSY_HAVE_RUNTIME_CPU
#include "../runtime/adapter_cpu.h"
#endif

#if TSY_HAVE_RUNTIME_CUDA
#include "../runtime/adapter_cuda.h"
#endif

namespace {

const char kUsage[] =
    "tsc — TensorSysY compiler\n"
    "\n"
    "Usage:\n"
    "  tsc <command> [options] <input.tsy>\n"
    "\n"
    "Commands:\n"
    "  parse     Parse the input file; exit 0 if the AST builds.\n"
    "  dump-ast  Parse and pretty-print the AST to stdout.\n"
    "  emit-hir  Parse, lower to HIR, run passes, print the MLIR-style dump.\n"
    "  emit-lir  Parse, lower to HIR+LIR, print the LIR dump.\n"
    "  emit-cpp  Generate a self-contained C++ host binary source.\n"
    "  emit-cu   Generate a self-contained CUDA .cu host binary source.\n"
    "  run-lir   Parse, lower, execute LIR with deterministic inputs.\n"
    "  --help    Show this message.\n"
    "\n"
    "Pipeline flags (apply to emit-hir / emit-lir / run-lir):\n"
    "  --opt=O0               Verify only (default).\n"
    "  --opt=O1               Verify + const-fold + dce + verify-post.\n"
    "  --disable-pass=<name>  Skip a pass by name (repeatable).\n"
    "\n"
    "run-lir backends:\n"
    "  --backend=native       (default) W4 naive in-tree kernels.\n"
    "  --backend=cpu-adapter  W6 mini-llm-engine/ops_cpu via runtime adapter.\n"
    "  --backend=cuda-adapter W8 self-written FP32 CUDA kernels + cuBLAS.\n"
    "\n"
    "emit-cpp output:\n"
    "  -o <path>              Write generated C++ to path instead of stdout.\n";

struct Options {
    std::string path;
    std::string opt = "O0";
    std::vector<std::string> disabled;
    std::string backend = "native";
    std::string output_path;  // empty = stdout
};

Options parseOptions(int argc, char** argv) {
    Options o;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        const std::string kOpt = "--opt=";
        const std::string kDis = "--disable-pass=";
        const std::string kBackend = "--backend=";
        const std::string kOutput = "--output=";
        if (a.rfind(kOpt, 0) == 0) {
            o.opt = a.substr(kOpt.size());
        } else if (a.rfind(kDis, 0) == 0) {
            o.disabled.push_back(a.substr(kDis.size()));
        } else if (a.rfind(kBackend, 0) == 0) {
            o.backend = a.substr(kBackend.size());
        } else if (a.rfind(kOutput, 0) == 0) {
            o.output_path = a.substr(kOutput.size());
        } else if (a == "-o") {
            if (i + 1 < argc) o.output_path = argv[++i];
        } else if (!a.empty() && a[0] != '-') {
            o.path = a;
        }
    }
    return o;
}

tsy::passes::PassManager buildPipeline(const Options& o) {
    auto pm = (o.opt == "O1") ? tsy::passes::buildPipelineO1()
                              : tsy::passes::buildPipelineO0();
    for (const auto& d : o.disabled) pm.disable(d);
    return pm;
}

int cmdParse(const std::string& path) {
    auto r = tsy::parseFile(path);
    if (!r.ok) {
        r.diagnostics.print(std::cerr);
        std::cerr << "parse failed: " << path << "\n";
        return 1;
    }
    std::cout << "parse ok: " << path << "\n";
    return 0;
}

int cmdDumpAst(const std::string& path) {
    auto r = tsy::parseFile(path);
    if (!r.ok) {
        r.diagnostics.print(std::cerr);
        std::cerr << "parse failed: " << path << "\n";
        return 1;
    }
    r.ast->dump(std::cout, 0);
    return 0;
}

// Parse + lower + run the requested pipeline. Shared preamble for
// emit-hir / emit-lir / run-lir. Returns null on any failure; diagnostics
// are already flushed to stderr before returning.
std::unique_ptr<tsy::hir::Module> parseAndRunPipeline(const Options& o,
                                                      tsy::DiagnosticEngine& diag) {
    auto r = tsy::parseFile(o.path);
    if (!r.ok) {
        r.diagnostics.print(std::cerr);
        std::cerr << "parse failed: " << o.path << "\n";
        return nullptr;
    }
    auto mod = tsy::hir::lowerAstToHir(*r.ast, r.diagnostics);
    if (!mod || r.diagnostics.hasErrors()) {
        r.diagnostics.print(std::cerr);
        std::cerr << "lowering failed: " << o.path << "\n";
        return nullptr;
    }
    auto pm = buildPipeline(o);
    pm.run(*mod, r.diagnostics);
    for (const auto& d : r.diagnostics.diagnostics()) {
        diag.report(d.level, d.loc, d.message);
    }
    if (r.diagnostics.hasErrors()) {
        std::cerr << "pipeline failed: " << o.path << "\n";
        return nullptr;
    }
    return mod;
}

int cmdEmitHir(const Options& o) {
    tsy::DiagnosticEngine diag;
    auto mod = parseAndRunPipeline(o, diag);
    if (!mod) return 1;
    tsy::hir::printModule(std::cout, *mod);
    return 0;
}

int cmdEmitCpp(const Options& o) {
    tsy::DiagnosticEngine diag;
    auto hmod = parseAndRunPipeline(o, diag);
    if (!hmod) return 1;
    auto lmod = tsy::lir::lowerHirToLir(*hmod, diag);
    if (!lmod || diag.hasErrors()) {
        diag.print(std::cerr);
        std::cerr << "lir lowering failed: " << o.path << "\n";
        return 1;
    }

    std::ostream* out = &std::cout;
    std::ofstream ofs;
    if (!o.output_path.empty()) {
        ofs.open(o.output_path);
        if (!ofs) {
            std::cerr << "emit-cpp: cannot write to '" << o.output_path << "'\n";
            return 1;
        }
        out = &ofs;
    }
    tsy::codegen::emitCppModule(*out, *lmod, o.path);
    return 0;
}

int cmdEmitCu(const Options& o) {
    tsy::DiagnosticEngine diag;
    auto hmod = parseAndRunPipeline(o, diag);
    if (!hmod) return 1;
    auto lmod = tsy::lir::lowerHirToLir(*hmod, diag);
    if (!lmod || diag.hasErrors()) {
        diag.print(std::cerr);
        std::cerr << "lir lowering failed: " << o.path << "\n";
        return 1;
    }

    std::ostream* out = &std::cout;
    std::ofstream ofs;
    if (!o.output_path.empty()) {
        ofs.open(o.output_path);
        if (!ofs) {
            std::cerr << "emit-cu: cannot write to '" << o.output_path << "'\n";
            return 1;
        }
        out = &ofs;
    }
    tsy::codegen::emitCudaModule(*out, *lmod, o.path);
    return 0;
}

int cmdEmitLir(const Options& o) {
    tsy::DiagnosticEngine diag;
    auto hmod = parseAndRunPipeline(o, diag);
    if (!hmod) return 1;
    auto lmod = tsy::lir::lowerHirToLir(*hmod, diag);
    if (!lmod || diag.hasErrors()) {
        diag.print(std::cerr);
        std::cerr << "lir lowering failed: " << o.path << "\n";
        return 1;
    }
    tsy::lir::printModule(std::cout, *lmod);
    return 0;
}

int cmdRunLir(const Options& o) {
    tsy::DiagnosticEngine diag;
    auto hmod = parseAndRunPipeline(o, diag);
    if (!hmod) return 1;
    auto lmod = tsy::lir::lowerHirToLir(*hmod, diag);
    if (!lmod || diag.hasErrors()) {
        diag.print(std::cerr);
        std::cerr << "lir lowering failed: " << o.path << "\n";
        return 1;
    }
    tsy::lir::RunResult result;
    if (o.backend == "cpu-adapter") {
#if TSY_HAVE_RUNTIME_CPU
        result = tsy::runtime::runWithCpuAdapter(*lmod, diag);
#else
        std::cerr << "run-lir: --backend=cpu-adapter requires tsy_runtime_cpu, "
                     "which was not built. Check mini-llm-engine path at CMake time.\n";
        return 1;
#endif
    } else if (o.backend == "cuda-adapter") {
#if TSY_HAVE_RUNTIME_CUDA
        result = tsy::runtime::runWithCudaAdapter(*lmod, diag);
#else
        std::cerr << "run-lir: --backend=cuda-adapter requires tsy_runtime_cuda, "
                     "which was not built. Check CUDA toolchain at CMake time.\n";
        return 1;
#endif
    } else if (o.backend == "native" || o.backend.empty()) {
        result = tsy::lir::runFirstTensorFunction(*lmod, diag);
    } else {
        std::cerr << "run-lir: unknown backend '" << o.backend << "'\n";
        return 1;
    }
    if (!result.ok || diag.hasErrors()) {
        diag.print(std::cerr);
        std::cerr << "run failed: " << o.path << "\n";
        return 1;
    }
    tsy::lir::printRunResult(std::cout, result);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << kUsage;
        return 1;
    }

    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        std::cout << kUsage;
        return 0;
    }

    if (argc < 3) {
        std::cerr << "tsc " << cmd << ": missing input file.\n\n" << kUsage;
        return 1;
    }
    auto opts = parseOptions(argc, argv);
    if (opts.path.empty()) {
        std::cerr << "tsc " << cmd << ": missing input file.\n\n" << kUsage;
        return 1;
    }

    if (cmd == "parse") return cmdParse(opts.path);
    if (cmd == "dump-ast") return cmdDumpAst(opts.path);
    if (cmd == "emit-hir") return cmdEmitHir(opts);
    if (cmd == "emit-lir") return cmdEmitLir(opts);
    if (cmd == "emit-cpp") return cmdEmitCpp(opts);
    if (cmd == "emit-cu") return cmdEmitCu(opts);
    if (cmd == "run-lir") return cmdRunLir(opts);

    std::cerr << "tsc: unknown command '" << cmd << "'\n\n" << kUsage;
    return 1;
}
