cmake_minimum_required(VERSION 3.18)
project(tensor-sysy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# --- tools ----------------------------------------------------------------

find_package(BISON 2.3 REQUIRED)
find_package(FLEX REQUIRED)

set(FRONTEND_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend)
set(FRONTEND_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/frontend_gen)
file(MAKE_DIRECTORY ${FRONTEND_GEN_DIR})

bison_target(TsyParser
    ${FRONTEND_SRC_DIR}/sysy.y
    ${FRONTEND_GEN_DIR}/sysy.tab.cpp
    DEFINES_FILE ${FRONTEND_GEN_DIR}/sysy.tab.hpp
)

flex_target(TsyLexer
    ${FRONTEND_SRC_DIR}/sysy.l
    ${FRONTEND_GEN_DIR}/sysy.lex.cpp
)

add_flex_bison_dependency(TsyLexer TsyParser)

# --- frontend library -----------------------------------------------------

add_library(tsy_frontend STATIC
    src/frontend/ast.cpp
    src/frontend/diagnostics.cpp
    src/frontend/parser_driver.cpp
    ${BISON_TsyParser_OUTPUTS}
    ${FLEX_TsyLexer_OUTPUTS}
)

target_include_directories(tsy_frontend PUBLIC
    ${FRONTEND_SRC_DIR}
    ${FRONTEND_GEN_DIR}
)

# --- HIR library ----------------------------------------------------------

add_library(tsy_hir STATIC
    src/hir/types.cpp
    src/hir/ops.cpp
    src/hir/printer.cpp
    src/hir/lowering.cpp
    src/hir/const_eval.cpp
    src/hir/verifier.cpp
)

target_include_directories(tsy_hir PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hir
)

target_link_libraries(tsy_hir PUBLIC tsy_frontend)

target_compile_options(tsy_hir PRIVATE -Wall -Wextra)

# --- Passes library -------------------------------------------------------

add_library(tsy_passes STATIC
    src/passes/pass_manager.cpp
    src/passes/verify.cpp
    src/passes/const_fold.cpp
    src/passes/dce.cpp
    src/passes/layout_lowering.cpp
    src/passes/schedule_cuda.cpp
)

target_include_directories(tsy_passes PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/passes
)

target_link_libraries(tsy_passes PUBLIC tsy_hir tsy_lir)

target_compile_options(tsy_passes PRIVATE -Wall -Wextra)

# --- LIR library ----------------------------------------------------------

add_library(tsy_lir STATIC
    src/lir/printer.cpp
    src/lir/lowering.cpp
    src/lir/interpreter.cpp
    src/lir/module_utils.cpp
)

target_include_directories(tsy_lir PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lir
)

target_link_libraries(tsy_lir PUBLIC tsy_hir)

target_compile_options(tsy_lir PRIVATE -Wall -Wextra)

# --- Codegen library ------------------------------------------------------

add_library(tsy_codegen STATIC
    src/codegen/cpp.cpp
    src/codegen/cuda.cpp
)

target_include_directories(tsy_codegen PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codegen
)

target_link_libraries(tsy_codegen PUBLIC tsy_lir)

target_compile_options(tsy_codegen PRIVATE -Wall -Wextra)

# --- Runtime adapter (CPU, calls mini-llm-engine/ops_cpu) ------------------
#
# Looks for mini-llm-engine next to this checkout by default; override with
#   -DTSY_MINI_LLM_ENGINE_SRC=/path/to/mini-llm-engine/mini-llm-engine/src
# when the sibling layout isn't available. If the source file can't be
# found, we skip tsy_runtime_cpu rather than fail configure — the W6
# tests are the only consumers and they gate themselves on the target.
if(NOT DEFINED TSY_MINI_LLM_ENGINE_SRC)
    set(TSY_MINI_LLM_ENGINE_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/../mini-llm-engine/mini-llm-engine/src"
        CACHE PATH "Path to mini-llm-engine/src (contains ops_cpu.cpp).")
endif()

if(EXISTS "${TSY_MINI_LLM_ENGINE_SRC}/ops_cpu.cpp")
    add_library(tsy_runtime_cpu STATIC
        src/runtime/adapter_cpu.cpp
        ${TSY_MINI_LLM_ENGINE_SRC}/ops_cpu.cpp
    )
    target_include_directories(tsy_runtime_cpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
        ${TSY_MINI_LLM_ENGINE_SRC}   # for #include "ops_cpu.h"
    )
    target_link_libraries(tsy_runtime_cpu PUBLIC tsy_lir)
    target_compile_options(tsy_runtime_cpu PRIVATE -Wall -Wextra)
    set(TSY_HAVE_RUNTIME_CPU 1)
    message(STATUS "tsy_runtime_cpu: bridging to ${TSY_MINI_LLM_ENGINE_SRC}")
else()
    set(TSY_HAVE_RUNTIME_CPU 0)
    message(WARNING
        "tsy_runtime_cpu: ${TSY_MINI_LLM_ENGINE_SRC}/ops_cpu.cpp not found; "
        "W6 adapter tests will be skipped. Set -DTSY_MINI_LLM_ENGINE_SRC to override.")
endif()

# Relax warnings on generated flex/bison sources; they produce noisy output
# under modern clang.
set_source_files_properties(
    ${BISON_TsyParser_OUTPUTS}
    ${FLEX_TsyLexer_OUTPUTS}
    PROPERTIES COMPILE_FLAGS "-Wno-unused-function -Wno-deprecated-register -Wno-register -Wno-sign-compare -Wno-unneeded-internal-declaration"
)

# Our own sources: warnings as usual, but not -Werror by default (iterate).
target_compile_options(tsy_frontend PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# --- Runtime adapter (CUDA, W8) ------------------------------------------
#
# Self-written FP32 kernels + cuBLAS sgemm for matmul. Independent of
# mini-llm-engine (their ops_cuda.cu is FP16-only). Gated at configure
# time so machines without CUDA still see 21/21 on W0-W7 tests.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

    # WSL2 libcuda stub fix: apt's libcuda.so shadows the real driver in
    # /usr/lib/wsl/lib/. DT_RPATH (not RUNPATH) propagates to dlopen inside
    # cudart. Borrowed verbatim from mini-llm-engine/CMakeLists.txt.
    if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
        set(CMAKE_BUILD_RPATH "/usr/lib/wsl/lib")
        add_link_options("LINKER:--disable-new-dtags")
        set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -L/usr/lib/wsl/lib -lcuda -Wl,--as-needed")
    endif()

    find_package(CUDAToolkit REQUIRED)

    add_library(tsy_runtime_cuda STATIC src/runtime/adapter_cuda.cu)
    target_include_directories(tsy_runtime_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime)
    target_link_libraries(tsy_runtime_cuda PUBLIC
        tsy_lir CUDA::cudart CUDA::cublas)
    target_compile_options(tsy_runtime_cuda PRIVATE -Wall -Wextra)

    set(TSY_HAVE_RUNTIME_CUDA 1)
    message(STATUS "tsy_runtime_cuda: enabled (arch sm_86)")
else()
    set(TSY_HAVE_RUNTIME_CUDA 0)
    message(STATUS "tsy_runtime_cuda: CUDA not detected; CUDA tests skipped")
endif()

# --- tsc executable -------------------------------------------------------

add_executable(tsc src/tools/tsc.cpp)
target_link_libraries(tsc PRIVATE tsy_passes tsy_codegen)
if(TSY_HAVE_RUNTIME_CPU)
    target_link_libraries(tsc PRIVATE tsy_runtime_cpu)  # brings tsy_lir
    target_compile_definitions(tsc PRIVATE TSY_HAVE_RUNTIME_CPU=1)
else()
    target_link_libraries(tsc PRIVATE tsy_lir)
endif()
if(TSY_HAVE_RUNTIME_CUDA)
    target_link_libraries(tsc PRIVATE tsy_runtime_cuda)
    target_compile_definitions(tsc PRIVATE TSY_HAVE_RUNTIME_CUDA=1)
endif()

# --- W7: helper to compile a .tsy example into a host binary --------------
#
# Runs tsc at build time to regenerate the .cpp sidecar, then drops it
# into a plain add_executable so the build system picks up incremental
# rebuilds. Only works when tsy_runtime_cpu is available (adapter is the
# codegen's runtime).
function(tsy_add_example NAME TSY_FILE)
    if(NOT TSY_HAVE_RUNTIME_CPU)
        message(STATUS "tsy_add_example(${NAME}): skipped (no tsy_runtime_cpu).")
        return()
    endif()
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${TSY_FILE}")
    set(_gen "${CMAKE_CURRENT_BINARY_DIR}/gen/${NAME}.cpp")
    add_custom_command(
        OUTPUT ${_gen}
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_BINARY_DIR}/gen
        COMMAND $<TARGET_FILE:tsc> emit-cpp ${_src} -o ${_gen}
        DEPENDS tsc ${_src}
        COMMENT "tsc emit-cpp ${TSY_FILE} -> gen/${NAME}.cpp"
        VERBATIM
    )
    add_executable(${NAME} ${_gen})
    target_link_libraries(${NAME} PRIVATE tsy_runtime_cpu)
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/out")
endfunction()

tsy_add_example(mlp examples/mlp.tsy)

# --- W8: helper to compile a .tsy example into a CUDA host binary --------
function(tsy_add_cuda_example NAME TSY_FILE)
    if(NOT TSY_HAVE_RUNTIME_CUDA)
        message(STATUS "tsy_add_cuda_example(${NAME}): skipped (no CUDA).")
        return()
    endif()
    set(_src "${CMAKE_CURRENT_SOURCE_DIR}/${TSY_FILE}")
    set(_gen "${CMAKE_CURRENT_BINARY_DIR}/gen/${NAME}.cu")
    add_custom_command(
        OUTPUT ${_gen}
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_BINARY_DIR}/gen
        COMMAND $<TARGET_FILE:tsc> emit-cu ${_src} -o ${_gen}
        DEPENDS tsc ${_src}
        COMMENT "tsc emit-cu ${TSY_FILE} -> gen/${NAME}.cu"
        VERBATIM
    )
    add_executable(${NAME} ${_gen})
    set_source_files_properties(${_gen} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${NAME} PRIVATE tsy_runtime_cuda)
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/out"
        CUDA_ARCHITECTURES 86)
endfunction()

tsy_add_cuda_example(matmul_cuda_demo examples/matmul_tiny_cuda.tsy)

if(TSY_HAVE_RUNTIME_CUDA)
    # W9: CUDA-event precision matmul bench binary.
    add_executable(tsy-bench src/tools/tsy-bench.cu)
    set_source_files_properties(src/tools/tsy-bench.cu PROPERTIES LANGUAGE CUDA)
    target_link_libraries(tsy-bench PRIVATE tsy_runtime_cuda)
    set_target_properties(tsy-bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CUDA_ARCHITECTURES 86)
endif()

# --- tests ----------------------------------------------------------------

enable_testing()
add_subdirectory(tests)
