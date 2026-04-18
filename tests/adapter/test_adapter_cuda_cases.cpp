// W8 adapter_cuda cases — parity with adapter_cpu on a handful of shapes.
// Each test runs the same primitive on the CPU adapter and on the CUDA
// adapter with identical inputs; both outputs must agree within
// atol=1e-4, rtol=1e-3 (FP32 cuBLAS vs FP32 CPU reference).

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/runtime/adapter_cpu.h"
#include "../../src/runtime/adapter_cuda.h"

using tsy::lir::NamedTensor;
using tsy::runtime::adapterAdd;
using tsy::runtime::adapterAddCuda;
using tsy::runtime::adapterMatMul;
using tsy::runtime::adapterMatMulCuda;
using tsy::runtime::adapterRMSNorm;
using tsy::runtime::adapterRMSNormCuda;
using tsy::runtime::adapterSoftmax;
using tsy::runtime::adapterSoftmaxCuda;

namespace {

int g_failures = 0;

void fail(const std::string& label, const std::string& why) {
    std::cerr << "FAIL[" << label << "]: " << why << "\n";
    ++g_failures;
}

bool closeEnough(float a, float b, float atol, float rtol) {
    float diff = std::fabs(a - b);
    float tol  = atol + rtol * std::fabs(b);
    return diff <= tol;
}

NamedTensor makeTensor(const std::string& name,
                       const std::vector<int64_t>& dims,
                       std::vector<float> data) {
    NamedTensor t;
    t.name = name;
    t.dims = dims;
    t.data = std::move(data);
    return t;
}

NamedTensor zeros(const std::string& name, const std::vector<int64_t>& dims) {
    NamedTensor t;
    t.name = name;
    t.dims = dims;
    int64_t n = 1;
    for (auto d : dims) n *= d;
    t.data.assign(n, 0.0f);
    return t;
}

std::vector<float> linspace(int64_t n, float start = 0.0f, float step = 0.1f) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = start + step * static_cast<float>(i);
    return v;
}

bool compareTensors(const std::string& label,
                    const NamedTensor& cpu, const NamedTensor& cuda,
                    float atol = 1e-4f, float rtol = 1e-3f) {
    if (cpu.data.size() != cuda.data.size()) {
        fail(label, "size mismatch: cpu=" + std::to_string(cpu.data.size()) +
                     " cuda=" + std::to_string(cuda.data.size()));
        return false;
    }
    for (size_t i = 0; i < cpu.data.size(); ++i) {
        if (!closeEnough(cuda.data[i], cpu.data[i], atol, rtol)) {
            std::ostringstream oss;
            oss << "idx=" << i << " cpu=" << cpu.data[i]
                << " cuda=" << cuda.data[i]
                << " diff=" << std::fabs(cuda.data[i] - cpu.data[i]);
            fail(label, oss.str());
            return false;
        }
    }
    return true;
}

void testMatmul(const std::string& label, int64_t M, int64_t K, int64_t N) {
    auto A = makeTensor("A", {M, K}, linspace(M * K, 0.0f, 0.1f));
    auto B = makeTensor("B", {K, N}, linspace(K * N, 0.5f, 0.1f));
    auto C_cpu  = zeros("C", {M, N});
    auto C_cuda = zeros("C", {M, N});
    adapterMatMul(A, B, C_cpu);
    adapterMatMulCuda(A, B, C_cuda);
    compareTensors(label, C_cpu, C_cuda);
}

void testAdd(const std::string& label,
             const std::vector<int64_t>& dims) {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    auto A = makeTensor("A", dims, linspace(n, 0.0f, 0.1f));
    auto B = makeTensor("B", dims, linspace(n, 1.0f, 0.2f));
    auto C_cpu  = zeros("C", dims);
    auto C_cuda = zeros("C", dims);
    adapterAdd(A, B, C_cpu);
    adapterAddCuda(A, B, C_cuda);
    compareTensors(label, C_cpu, C_cuda);
}

void testSoftmax(const std::string& label,
                 const std::vector<int64_t>& dims) {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    auto X = makeTensor("X", dims, linspace(n, -1.0f, 0.3f));
    auto Y_cpu  = zeros("Y", dims);
    auto Y_cuda = zeros("Y", dims);
    adapterSoftmax(X, Y_cpu);
    adapterSoftmaxCuda(X, Y_cuda);
    compareTensors(label, Y_cpu, Y_cuda);
}

void testRMSNorm(const std::string& label,
                 const std::vector<int64_t>& dims) {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    auto X = makeTensor("X", dims, linspace(n, 0.5f, 0.25f));
    auto Y_cpu  = zeros("Y", dims);
    auto Y_cuda = zeros("Y", dims);
    adapterRMSNorm(X, Y_cpu);
    adapterRMSNormCuda(X, Y_cuda);
    compareTensors(label, Y_cpu, Y_cuda);
}

void testMatmulVariant(const std::string& label,
                        int64_t M, int64_t K, int64_t N,
                        const std::string& variant) {
    auto A = makeTensor("A", {M, K}, linspace(M * K, 0.0f, 0.1f));
    auto B = makeTensor("B", {K, N}, linspace(K * N, 0.5f, 0.1f));
    auto C_cpu  = zeros("C", {M, N});
    auto C_cuda = zeros("C", {M, N});
    adapterMatMul(A, B, C_cpu);
    adapterMatMulCuda(A, B, C_cuda, variant);
    compareTensors(label, C_cpu, C_cuda);
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    // MatMul — 5 shape classes (M, K, N)
    testMatmul("matmul/square-8",    8,   8,  8);
    testMatmul("matmul/tall-128x16", 128, 16, 8);
    testMatmul("matmul/odd-7x13",    7,   13, 11);
    testMatmul("matmul/1xK-row",     1,   32, 8);
    testMatmul("matmul/Kx1-col",     8,   32, 1);

    // Add / Softmax / RMSNorm — 3 shape classes
    for (auto dims : std::vector<std::vector<int64_t>>{
             {8, 8}, {7, 13}, {1, 32}}) {
        std::string label = "shape-" + std::to_string(dims[0]) + "x"
                                      + std::to_string(dims[1]);
        testAdd    ("add/"     + label, dims);
        testSoftmax("softmax/" + label, dims);
        testRMSNorm("rmsnorm/" + label, dims);
    }

    // W9 per-variant parity.
    // 64x64x64: exercises naive + cublas (tiled requires M%128==0).
    // 128x128x128: exercises all three including tiled.
    testMatmulVariant("matmul/64x64x64/naive",    64,  64,  64, "naive");
    testMatmulVariant("matmul/64x64x64/cublas",   64,  64,  64, "cublas");
    testMatmulVariant("matmul/128x128x128/naive", 128, 128, 128, "naive");
    testMatmulVariant("matmul/128x128x128/tiled", 128, 128, 128, "tiled");
    testMatmulVariant("matmul/128x128x128/cublas",128, 128, 128, "cublas");

    if (g_failures == 0) {
        std::cout << "adapter_cuda_cases: ALL PASS\n";
        return 0;
    }
    std::cerr << "adapter_cuda_cases: " << g_failures << " failure(s)\n";
    return 1;
}
