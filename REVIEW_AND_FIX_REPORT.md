# Review and Fix Report

## Changes
- Aligned HIR verifier with runtime capability by limiting matmul to rank-2 operands.
- Added verifier checks that resolved tensor dimensions must be positive.
- Added fixtures and shape tests for rank-3 matmul, zero dimension, and negative dimension diagnostics.
- Made e2e and benchmark helpers use explicit build artifact paths via env/CLI instead of hardcoding source `build/`.
- Adjusted CTest registration to pass real build paths and tolerate missing Python numpy/pytest by not registering e2e tests.
- Removed Bison syntax unsupported by the system Bison used during local verification.

## Verification
- `cmake -S . -B /tmp/tensor-sysy-build -DCMAKE_BUILD_TYPE=Debug` passed.
- `cmake --build /tmp/tensor-sysy-build -j` passed.
- `ctest --test-dir /tmp/tensor-sysy-build --output-on-failure` passed with 26/26 tests.
- `git diff --check` passed.

## Remaining
- CUDA-specific tests were skipped because CUDA was not detected.
- Python e2e pytest was not registered because the selected system Python lacks numpy/pytest.
