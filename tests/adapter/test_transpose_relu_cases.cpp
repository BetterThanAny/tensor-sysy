// W10 unit tests for the two new primitives added to the CPU adapter.
// Follows the same "g_failures + fail() + tiny helpers" pattern as
// test_adapter_cpu_cases.cpp and test_adapter_cuda_cases.cpp.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/runtime/adapter_cpu.h"

using tsy::lir::NamedTensor;
using tsy::runtime::adapterReLU;
using tsy::runtime::adapterTranspose;

namespace {

int g_failures = 0;

void fail(const std::string& label, const std::string& why) {
    std::cerr << "FAIL[" << label << "]: " << why << "\n";
    ++g_failures;
}

bool approx(float a, float b, float atol = 1e-6f) {
    return std::fabs(a - b) <= atol;
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

NamedTensor zeros(const std::string& name,
                  const std::vector<int64_t>& dims) {
    NamedTensor t;
    t.name = name;
    t.dims = dims;
    int64_t n = 1;
    for (auto d : dims) n *= d;
    t.data.assign(n, 0.0f);
    return t;
}

void checkVec(const std::string& label, const NamedTensor& t,
              const std::vector<float>& expected) {
    if (t.data.size() != expected.size()) {
        fail(label, "size mismatch: got " + std::to_string(t.data.size()) +
                     ", expected " + std::to_string(expected.size()));
        return;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!approx(t.data[i], expected[i])) {
            std::ostringstream oss;
            oss << "idx=" << i << " got " << t.data[i]
                << " expected " << expected[i];
            fail(label, oss.str());
            return;
        }
    }
}

void testTranspose2x2() {
    auto x = makeTensor("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto y = zeros("y", {2, 2});
    adapterTranspose(x, y);
    checkVec("transpose/2x2", y, {1.0f, 3.0f, 2.0f, 4.0f});
}

void testTranspose2x3() {
    // input  shape [2, 3]:  [[1, 2, 3], [4, 5, 6]]
    // output shape [3, 2]:  [[1, 4], [2, 5], [3, 6]]
    auto x = makeTensor("x", {2, 3}, {1, 2, 3, 4, 5, 6});
    auto y = zeros("y", {3, 2});
    adapterTranspose(x, y);
    checkVec("transpose/2x3", y, {1, 4, 2, 5, 3, 6});
}

void testReluFlat() {
    auto x = makeTensor("x", {4}, {-1.0f, 0.0f, 1.0f, 2.0f});
    auto y = zeros("y", {4});
    adapterReLU(x, y);
    checkVec("relu/flat", y, {0.0f, 0.0f, 1.0f, 2.0f});
}

void testRelu2D() {
    auto x = makeTensor("x", {2, 3},
                        {-0.5f, 0.0f, 0.5f, 1.0f, -1.0f, 2.0f});
    auto y = zeros("y", {2, 3});
    adapterReLU(x, y);
    checkVec("relu/2x3", y, {0.0f, 0.0f, 0.5f, 1.0f, 0.0f, 2.0f});
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    testTranspose2x2();
    testTranspose2x3();
    testReluFlat();
    testRelu2D();
    if (g_failures == 0) {
        std::cout << "transpose_relu_cases: ALL PASS\n";
        return 0;
    }
    std::cerr << "transpose_relu_cases: " << g_failures << " failure(s)\n";
    return 1;
}
