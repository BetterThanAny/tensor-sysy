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

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    testMatmul("matmul/2x2x2", 2, 2, 2);

    if (g_failures == 0) {
        std::cout << "adapter_cuda_cases: ALL PASS\n";
        return 0;
    }
    std::cerr << "adapter_cuda_cases: " << g_failures << " failure(s)\n";
    return 1;
}
