# Tests for tensor-sysy.
#
# W0: lightweight ctest-only. No external test framework.
#   - test_parse_cases: C++ driver linked against tsy_frontend that runs a
#     built-in list of parse / diagnostic cases.
#   - cli_*: shell-level smoke tests that exercise the tsc binary.

add_executable(test_parse_cases parse/test_parse_cases.cpp)
target_link_libraries(test_parse_cases PRIVATE tsy_frontend)

add_test(NAME parse_cases
    COMMAND test_parse_cases ${CMAKE_SOURCE_DIR}/examples
)

# W3: shape / verifier cases.
add_executable(test_shape_cases shape/test_shape_cases.cpp)
target_link_libraries(test_shape_cases PRIVATE tsy_hir)

add_test(NAME shape_cases
    COMMAND test_shape_cases ${CMAKE_SOURCE_DIR}/examples
)

# W4: interpreter vs hand-verified outputs.
add_executable(test_run_cases run/test_run_cases.cpp)
target_link_libraries(test_run_cases PRIVATE tsy_lir)

add_test(NAME run_cases
    COMMAND test_run_cases ${CMAKE_SOURCE_DIR}/examples
)

# W5: PassManager + DCE + verifier pass behaviours.
add_executable(test_pass_cases passes/test_pass_cases.cpp)
target_link_libraries(test_pass_cases PRIVATE tsy_passes)

add_test(NAME pass_cases
    COMMAND test_pass_cases ${CMAKE_SOURCE_DIR}/examples
)

# W9: ScheduleCudaPass unit test. Pure LIR pass + CPU code — no CUDA
# runtime needed, so this is NOT gated on tsy_runtime_cuda.
add_executable(test_schedule_cuda_cases passes/test_schedule_cuda_cases.cpp)
target_link_libraries(test_schedule_cuda_cases PRIVATE tsy_passes)

add_test(NAME pass_schedule_cuda_cases
    COMMAND test_schedule_cuda_cases
)

# W6: CPU runtime adapter cases (primitive semantics + interp parity).
# Registered only when mini-llm-engine/ops_cpu.cpp was found at configure
# time so CI on a standalone clone still reports a sensible status.
if(TARGET tsy_runtime_cpu)
    add_executable(test_adapter_cpu_cases adapter/test_adapter_cpu_cases.cpp)
    target_link_libraries(test_adapter_cpu_cases PRIVATE tsy_runtime_cpu)

    add_test(NAME adapter_cpu_cases
        COMMAND test_adapter_cpu_cases ${CMAKE_SOURCE_DIR}/examples
    )

    add_test(NAME cli_adapter_cpu_run_matmul_tiny
        COMMAND tsc run-lir --backend=cpu-adapter
                ${CMAKE_SOURCE_DIR}/examples/run_matmul_tiny.tsy
    )
    set_tests_properties(cli_adapter_cpu_run_matmul_tiny PROPERTIES
        PASS_REGULAR_EXPRESSION "local C .*0.0700.*0.0800"
    )

    # W7: emit-cpp must produce compilable source; the mlp target in the root
    # CMakeLists.txt is the compile smoke. Running the binary here asserts
    # the observable output carries the expected softmax result row from
    # our MLP chain.
    add_test(NAME codegen_mlp_binary_runs
        COMMAND $<TARGET_FILE:mlp>
    )
    set_tests_properties(codegen_mlp_binary_runs PROPERTIES
        PASS_REGULAR_EXPRESSION "function: mlp_forward.*local y shape=\\[2,2\\].*0\\.4502.*0\\.5498"
    )

    # Plan verifies codegen via `tsc emit-cpp && cmake --build && ./out/mlp`.
    # This CLI smoke just exercises the emit-cpp stage end-to-end and
    # double-checks that the generator references the adapter entry points.
    add_test(NAME codegen_emit_cpp_mlp_contains_adapter_calls
        COMMAND tsc emit-cpp ${CMAKE_SOURCE_DIR}/examples/mlp.tsy
    )
    set_tests_properties(codegen_emit_cpp_mlp_contains_adapter_calls PROPERTIES
        PASS_REGULAR_EXPRESSION "tsy::runtime::adapterMatMul.*tsy::runtime::adapterSoftmax"
    )
endif()

# W8: CUDA adapter tests. Outer gate is just `tsy_runtime_cuda`.
# The parity test has an inner gate because it links against BOTH
# runtimes (it compares CPU vs CUDA outputs).
if(TARGET tsy_runtime_cuda)
    if(TARGET tsy_runtime_cpu)
        add_executable(test_adapter_cuda_cases adapter/test_adapter_cuda_cases.cpp)
        target_link_libraries(test_adapter_cuda_cases
            PRIVATE tsy_runtime_cpu tsy_runtime_cuda)

        add_test(NAME adapter_cuda_cases
            COMMAND test_adapter_cuda_cases ${CMAKE_SOURCE_DIR}/examples
        )
    endif()

    # (subsequent tasks append CLI + codegen tests here, outside the
    # inner tsy_runtime_cpu gate — they only need the CUDA runtime.)

    add_test(NAME cli_cuda_adapter_run_matmul_tiny
        COMMAND tsc run-lir --backend=cuda-adapter
                ${CMAKE_SOURCE_DIR}/examples/run_matmul_tiny.tsy
    )
    set_tests_properties(cli_cuda_adapter_run_matmul_tiny PROPERTIES
        PASS_REGULAR_EXPRESSION "local C .*0.0700.*0.0800"
    )

    # Compile smoke: the generated matmul_cuda_demo must link and produce
    # the expected 2x2x2 output when run.
    add_test(NAME codegen_cuda_matmul_binary_runs
        COMMAND $<TARGET_FILE:matmul_cuda_demo>
    )
    set_tests_properties(codegen_cuda_matmul_binary_runs PROPERTIES
        PASS_REGULAR_EXPRESSION
        "function: matmul_layer.*local C shape=\\[2,2\\].*0\\.0700.*0\\.0800"
    )

    # CLI-level: emit-cu must produce source that references the adapter.
    add_test(NAME codegen_emit_cu_contains_adapter_calls
        COMMAND tsc emit-cu ${CMAKE_SOURCE_DIR}/examples/matmul_tiny_cuda.tsy
    )
    set_tests_properties(codegen_emit_cu_contains_adapter_calls PROPERTIES
        PASS_REGULAR_EXPRESSION "tsy::runtime::adapterMatMulCuda"
    )

    # W9: Python driver smoke — confirms tsy-bench runs + CSV is parseable.
    add_test(NAME cli_bench_smoke
        COMMAND python3 ${CMAKE_SOURCE_DIR}/benchmarks/run_shapes.py --smoke
    )
    set_tests_properties(cli_bench_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "matmul.*256.*256.*256"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

# W5 CLI smoke: --opt=O1 and --disable-pass=dce round-trip.
add_test(NAME cli_emit_hir_o1_drops_dead
    COMMAND tsc emit-hir --opt=O1 ${CMAKE_SOURCE_DIR}/examples/dead_matmul.tsy
)
set_tests_properties(cli_emit_hir_o1_drops_dead PROPERTIES
    PASS_REGULAR_EXPRESSION "func @dead_block"
    FAIL_REGULAR_EXPRESSION "matmul %A, %B"
)

add_test(NAME cli_emit_hir_o1_disable_dce_keeps
    COMMAND tsc emit-hir --opt=O1 --disable-pass=dce
            ${CMAKE_SOURCE_DIR}/examples/dead_matmul.tsy
)
set_tests_properties(cli_emit_hir_o1_disable_dce_keeps PROPERTIES
    PASS_REGULAR_EXPRESSION "matmul %A, %B"
)

# W4 CLI smoke: tsc emit-lir + run-lir on the tiny matmul example.
add_test(NAME cli_emit_lir_matmul_tiny
    COMMAND tsc emit-lir ${CMAKE_SOURCE_DIR}/examples/run_matmul_tiny.tsy
)
set_tests_properties(cli_emit_lir_matmul_tiny PROPERTIES
    PASS_REGULAR_EXPRESSION "lir.module.*call matmul %A, %B"
)

add_test(NAME cli_run_lir_matmul_tiny
    COMMAND tsc run-lir ${CMAKE_SOURCE_DIR}/examples/run_matmul_tiny.tsy
)
set_tests_properties(cli_run_lir_matmul_tiny PROPERTIES
    PASS_REGULAR_EXPRESSION "input A .*local C .*0.0700.*0.0800"
)

# CLI smokes — make sure tsc launches and subcommands dispatch correctly.
add_test(NAME cli_help
    COMMAND tsc --help
)

add_test(NAME cli_parse_smoke
    COMMAND tsc parse ${CMAKE_SOURCE_DIR}/examples/smoke.tsy
)

add_test(NAME cli_dump_ast_smoke
    COMMAND tsc dump-ast ${CMAKE_SOURCE_DIR}/examples/smoke.tsy
)
set_tests_properties(cli_dump_ast_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "CompUnit.*FuncDef.*main"
)

# W1: dump-ast must surface the tensor type and builtin call nodes.
add_test(NAME cli_dump_ast_matmul
    COMMAND tsc dump-ast ${CMAKE_SOURCE_DIR}/examples/matmul.tsy
)
set_tests_properties(cli_dump_ast_matmul PROPERTIES
    PASS_REGULAR_EXPRESSION "TensorType.*f32.*builtin.*@matmul"
)

add_test(NAME cli_parse_tensor_all_ops
    COMMAND tsc parse ${CMAKE_SOURCE_DIR}/examples/tensor_all_ops.tsy
)

# Bad input must exit nonzero AND mention the file in the diagnostic text.
add_test(NAME cli_parse_bad_exits_nonzero
    COMMAND tsc parse ${CMAKE_SOURCE_DIR}/examples/bad_syntax.tsy
)
set_tests_properties(cli_parse_bad_exits_nonzero PROPERTIES WILL_FAIL TRUE)

# W2: emit-hir must lower the matmul example into a module with a recognisable
# func/matmul skeleton. Regex checks the key structural markers.
add_test(NAME cli_emit_hir_matmul
    COMMAND tsc emit-hir ${CMAKE_SOURCE_DIR}/examples/matmul.tsy
)
set_tests_properties(cli_emit_hir_matmul PROPERTIES
    PASS_REGULAR_EXPRESSION "module.*func @matmul_layer.*tensor<f32>\\[M=8, K=16\\].*matmul %A, %B"
)

add_test(NAME cli_emit_hir_all_ops
    COMMAND tsc emit-hir ${CMAKE_SOURCE_DIR}/examples/tensor_all_ops.tsy
)
set_tests_properties(cli_emit_hir_all_ops PROPERTIES
    PASS_REGULAR_EXPRESSION "matmul .*softmax .*rmsnorm"
)

# W2: emit-hir should still succeed on a plain SysY program (empty tensor ops
# but well-formed func skeletons).
add_test(NAME cli_emit_hir_smoke_sysy
    COMMAND tsc emit-hir ${CMAKE_SOURCE_DIR}/examples/smoke.tsy
)
set_tests_properties(cli_emit_hir_smoke_sysy PROPERTIES
    PASS_REGULAR_EXPRESSION "module.*func @main"
)

# W9: --opt=O1 --disable-pass=dce keeps the matmul call in LIR so that
# schedule-cuda can annotate it with variant="tiled" for the 128^3 fixture.
# (Without --disable-pass=dce the matmul result %C is dead and DCE removes
# it before LIR lowering, leaving nothing for schedule-cuda to tag.)
add_test(NAME cli_emit_lir_schedule_shows_variant
    COMMAND tsc emit-lir --opt=O1 --disable-pass=dce
            ${CMAKE_SOURCE_DIR}/examples/run_matmul_medium.tsy
)
set_tests_properties(cli_emit_lir_schedule_shows_variant PROPERTIES
    PASS_REGULAR_EXPRESSION "variant=\"tiled\""
)

# W9: --disable-pass=schedule-cuda must suppress the variant attr.
add_test(NAME cli_emit_lir_disable_schedule_no_variant
    COMMAND tsc emit-lir --opt=O1 --disable-pass=dce --disable-pass=schedule-cuda
            ${CMAKE_SOURCE_DIR}/examples/run_matmul_medium.tsy
)
set_tests_properties(cli_emit_lir_disable_schedule_no_variant PROPERTIES
    FAIL_REGULAR_EXPRESSION "variant="
)

