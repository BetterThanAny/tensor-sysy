// tsy-bench — CUDA-event precision matmul benchmark.
//
// Usage:
//   tsy-bench                    # default 5 shape × 3 variant = 15 rows
//   tsy-bench --smoke            # single shape × all variants (ctest use)
//   tsy-bench --shapes 1024x1024x1024  # single shape
//   tsy-bench --variants naive,tiled   # variant filter
//
// Output: CSV on stdout with header
//   primitive,M,K,N,variant,ms_median,gflops
//
// Timing: 3 warmup + 5 measured runs via cudaEvent, take the median.

#include "../runtime/adapter_cuda.h"
#include "../lir/interpreter.h"  // NamedTensor / fillDeterministic

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Shape { int M, K, N; };

std::vector<Shape> parseShapes(const std::string& csv_spec) {
    std::vector<Shape> out;
    std::stringstream ss(csv_spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        Shape s;
        if (std::sscanf(item.c_str(), "%dx%dx%d", &s.M, &s.K, &s.N) == 3) {
            out.push_back(s);
        }
    }
    return out;
}

std::vector<std::string> parseVariants(const std::string& csv_spec) {
    std::vector<std::string> out;
    std::stringstream ss(csv_spec);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

tsy::lir::NamedTensor makeBuf(const std::string& name,
                               const std::vector<int64_t>& dims) {
    tsy::lir::NamedTensor t;
    t.name = name;
    t.dims = dims;
    int64_t n = 1; for (auto d : dims) n *= d;
    t.data.assign(n, 0.0f);
    return t;
}

float medianMs(std::vector<float> ms) {
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

float benchOne(int M, int K, int N, const std::string& variant) {
    auto A = makeBuf("A", {M, K});
    auto B = makeBuf("B", {K, N});
    auto C = makeBuf("C", {M, N});
    tsy::lir::fillDeterministic(A, 0);
    tsy::lir::fillDeterministic(B, 1);

    // Warmup (3 runs, results discarded).
    for (int i = 0; i < 3; i++) {
        tsy::runtime::adapterMatMulCuda(A, B, C, variant);
    }
    cudaDeviceSynchronize();

    // Measured (5 runs).
    std::vector<float> times;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(t0);
        tsy::runtime::adapterMatMulCuda(A, B, C, variant);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        times.push_back(ms);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return medianMs(times);
}

float gflops(int M, int K, int N, float ms) {
    if (ms <= 0.0f) return 0.0f;
    const double ops = 2.0 * static_cast<double>(M) *
                              static_cast<double>(K) *
                              static_cast<double>(N);
    return static_cast<float>(ops / (static_cast<double>(ms) * 1e6));
}

struct Options {
    bool smoke = false;
    std::string shapes_arg;
    std::string variants_arg;
};

int runMatmulBench(const Options& opts) {
    std::vector<Shape> shapes;
    if (!opts.shapes_arg.empty()) shapes = parseShapes(opts.shapes_arg);
    else if (opts.smoke)          shapes = { {256, 256, 256} };
    else                          shapes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {128, 16, 8},
        {7, 13, 11},
    };

    std::vector<std::string> variants;
    if (!opts.variants_arg.empty()) variants = parseVariants(opts.variants_arg);
    else                            variants = { "naive", "tiled", "cublas" };

    std::cout << "primitive,M,K,N,variant,ms_median,gflops\n";
    for (const auto& s : shapes) {
        for (const auto& v : variants) {
            // Tiled kernel requires aligned dims; skip cleanly if not.
            if (v == "tiled") {
                if (s.M % 128 != 0 || s.N % 128 != 0 || s.K % 8 != 0) {
                    continue;
                }
            }
            float ms = benchOne(s.M, s.K, s.N, v);
            float gf = gflops(s.M, s.K, s.N, ms);
            std::cout << "matmul," << s.M << "," << s.K << "," << s.N << ","
                      << v << "," << ms << "," << gf << "\n";
        }
    }
    return 0;
}

int usage(const char* progname) {
    std::cerr << "usage: " << progname
              << " [--primitive matmul|transformer_block] [--smoke] "
              << "[--shapes MxKxN[,...]] [--variants v1[,v2,...]]\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    std::string primitive = "matmul";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--smoke") opts.smoke = true;
        else if (a.rfind("--primitive=", 0) == 0) primitive = a.substr(12);
        else if (a == "--primitive" && i + 1 < argc) primitive = argv[++i];
        else if (a.rfind("--shapes=", 0) == 0) opts.shapes_arg = a.substr(9);
        else if (a == "--shapes" && i + 1 < argc) opts.shapes_arg = argv[++i];
        else if (a.rfind("--variants=", 0) == 0) opts.variants_arg = a.substr(11);
        else if (a == "--variants" && i + 1 < argc) opts.variants_arg = argv[++i];
        else if (a == "-h" || a == "--help") { return usage(argv[0]); }
        else { return usage(argv[0]); }
    }

    if (primitive == "matmul") return runMatmulBench(opts);
    // Task 7 will add: if (primitive == "transformer_block") return runTransformerBlockBench();
    std::cerr << "unknown --primitive: " << primitive
              << " (valid: matmul)\n";
    return 2;
}
