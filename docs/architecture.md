# TensorSysY — Architecture

This document describes the pipeline stages, IR contracts, and adapter
boundary as of W11. It complements the 12-week plan in
[`../PLAN.md`](../PLAN.md).

## 1. Pipeline at a glance

```
  ┌─────────┐   flex/bison    ┌─────┐   AST→HIR      ┌──────┐
  │ .tsy    │ ──────────────► │ AST │ ────────────► │ HIR  │
  │ source  │   (sysy.l/y)    │     │  (lowering)   │      │
  └─────────┘                 └─────┘               └──┬───┘
                                                       │
                             HIR passes (verify, const-fold, dce)
                                                       │
                                                       ▼
                                                    ┌──────┐
                                                    │ HIR' │
                                                    └──┬───┘
                                                       │  hir→lir lowering
                                                       ▼
                                                    ┌──────┐
                                                    │ LIR  │
                                                    └──┬───┘
                                                       │
                                  LIR passes (layout-lowering, schedule-cuda)
                                                       │
                    ┌──────────────────┬───────────────┴─────────────┐
                    ▼                  ▼                             ▼
           ┌────────────────┐ ┌────────────────┐          ┌────────────────┐
           │ LIR interpreter│ │  C++ codegen   │          │  CUDA codegen  │
           │  (reference)   │ │  (emit-cpp)    │          │  (emit-cu)     │
           └───────┬────────┘ └────────┬───────┘          └────────┬───────┘
                   │                   │                           │
                   │                   ▼                           ▼
                   │           ┌────────────────┐          ┌────────────────┐
                   │           │  adapter_cpu   │          │  adapter_cuda  │
                   │           └────────┬───────┘          └────────┬───────┘
                   │                    │                           │
                   │                    ▼                           ▼
                   │            mini-llm-engine/ops_cpu      mini-llm-engine/ops_cuda
                   │
                   └──► three-way comparison (L5 triangle):
                        interpreter  ≡  CPU binary output  ≈  CUDA binary output
```

Tolerances for the triangle:

- interpreter vs CPU binary: `atol=1e-6` (identical math, just codegened)
- CUDA vs CPU: `atol=1e-4, rtol=1e-3` (cublas heuristics + fp32 reduction order)

## 2. Stage contracts

### 2.1 Frontend — `src/frontend/`

- `sysy.l` + `sysy.y` extend the original SysY grammar with:
  - `tensor<dtype>[dim, dim, ...]` type syntax.
  - Builtin operators invoked as `@matmul(A, B)`, `@add(X, Y)`,
    `@softmax(X)`, `@rmsnorm(X)`, `@transpose(X)`, `@relu(X)`.
- Every AST node carries a `SourceLocation` (line + column). Diagnostics
  produced by the verifier reuse the same engine as the parser — one
  path for "position → message".
- The original SysY regression suite is retained under `tests/parse/`
  (per W1 risk mitigation in PLAN).

### 2.2 HIR — `src/hir/`

Shape-typed, SSA-like ops. Relevant `OpKind`:

| Kind        | Purpose                                                                |
|-------------|------------------------------------------------------------------------|
| `MatMul`    | `[M,K] × [K,N] → [M,N]` (no implicit transpose; layout is contractual) |
| `Add`       | Elementwise; broadcast rules explicit, documented in adapter tests     |
| `Softmax`   | Along the **last** dimension                                           |
| `RMSNorm`   | Along the **hidden** dim (last)                                        |
| `Transpose` | 2-D only; `[d0, d1] → [d1, d0]`                                        |
| `ReLU`      | Elementwise, shape-preserving                                          |
| `FuncCall`  | Non-builtin call — preserved through HIR unchanged                     |
| `Return`    | Function terminator                                                    |
| `View`/`Permute` | Reserved; grammar parses but currently rejected in verifier       |

Each `Op` carries operands, results, `SourceLocation`, and a fallback
`builtin_name` so the printer can render unknown/unlowered AST shapes
without losing fidelity.

The **verifier** (`hir/verifier.cpp`) enforces shape/type invariants and
emits diagnostics with the original source location. The **lowerer**
(`hir/lowering.cpp`) turns AST into HIR — its failures are also
diagnostics, not assertions.

### 2.3 LIR — `src/lir/`

Loop-level IR: `buffers`, `loads`, `stores`, `calls`, and scheduling
variants. Two concrete consumers:

- **Interpreter** (`lir/interpreter.cpp`): naive but complete —
  deterministic inputs, no vendor calls. Exists so codegen bugs can't
  hide behind "numpy and PyTorch agree therefore we agree".
- **Codegen** (`src/codegen/`): emits either self-contained C++
  (linked against `adapter_cpu`) or CUDA `.cu` (linked against
  `adapter_cuda`). Code shape is the same across the two backends —
  the adapter resolves the actual kernel.

### 2.4 Passes — `src/passes/`

PassManager keeps two ordered vectors: HIR passes and LIR passes.
Pipeline definitions live in `pass_manager.cpp`:

```text
O0 (HIR):  verify
O1 (HIR):  verify → const-fold → dce → verify-post
   (LIR):  layout-lowering → schedule-cuda     (always appended)
```

Every pass has a stable name; `--disable-pass=<name>` skips it. This is
both a dev knob and a test vector — L4 requires each pass to be:

1. Structurally correct.
2. Semantics-preserving (compare output with and without the pass).
3. Idempotent (running twice == running once).
4. Disablable via `--disable-pass=X`.

- `const-fold` is currently a named placeholder — the W2–W5 milestone
  defined the pipeline slot and test contract; payloads ship as ops
  begin to have foldable patterns. This is intentional: having the
  slot stable early keeps downstream tests grounded.
- `layout-lowering` inserts explicit layout transforms when an operand
  arrives in a form the adapter doesn't accept, rather than letting
  the adapter silently materialise a copy.
- `schedule-cuda` picks a `variant` (naive / tiled / cublas) per
  matmul LIR node. `emit-lir` surfaces the variant in the dump — this
  is what `cli_emit_lir_schedule_shows_variant` and its `--disable-pass`
  counterpart test.

### 2.5 Runtime adapter — `src/runtime/`

The adapter is the **only** place allowed to call into
`mini-llm-engine`. This was the W6 risk call: keep HIR/LIR free of any
concrete kernel API so the backend stays swappable.

- `adapter_cpu.{h,cpp}` — calls `ops_cpu.*`.
- `adapter_cuda.{h,cu}` — calls `ops_cuda.*`, owns CUDA contexts,
  exposes explicit `sync()` points at result boundaries (W10 reviewer
  follow-up).

Semantics locked by `tests/adapter/`:

- MatMul layout is `[M,K] × [K,N]` — transposed variants must go through
  an explicit `@transpose` op; the adapter does NOT auto-transpose.
- Softmax is **last-dim**; any other axis is a verifier error upstream.
- RMSNorm hidden-dim match is checked upstream; the adapter asserts.
- Add broadcast rules follow mini-llm-engine's, documented via tests —
  see `test_adapter_cpu_cases.cpp` and `test_transpose_relu_cases.cpp`.
- Aliasing: the adapter never writes into a still-live SSA value
  (guarded by LIR scheduling + L3 aliasing tests).

### 2.6 Codegen — `src/codegen/`

Emits host code that:

1. Includes the right adapter header.
2. Declares input buffers with deterministic fill (matches the
   interpreter's deterministic-input rule — required for L5).
3. Calls the adapter in the LIR-scheduled order.
4. Prints tensors at the function boundaries so ctest can diff them.

Codegen is deliberately **dumb**: one LIR call becomes one adapter
call. Fusion and kernel selection belong in LIR passes (currently:
`schedule-cuda`).

## 3. Testing layers — how they exercise this

| Layer | Stage covered                              | Representative tests                                |
|-------|--------------------------------------------|-----------------------------------------------------|
| L0    | Build + CLI smoke                          | `cli_help`, `cli_bench_smoke`                       |
| L1    | AST / HIR / LIR / diag text                | golden/ tests via `test_run_cases`                  |
| L2    | Frontend + verifier pos/neg                | `cli_parse_*`, `shape_cases`, `bad_*.tsy`           |
| L3    | Adapter semantics                          | `adapter_cpu_cases`, `adapter_cuda_cases`, `transpose_relu_cases` |
| L4    | Pass pipeline                              | `pass_cases`, `pass_schedule_cuda_cases`, `cli_emit_hir_o1_*` |
| L5    | Three-way interp ↔ CPU codegen ↔ CUDA      | `codegen_mlp_binary_runs`, `codegen_cuda_matmul_binary_runs`, `e2e_transformer_block_pytest` |
| L6    | E2E vs PyTorch / numpy                     | `tests/e2e/test_transformer_block.py` (compares three backends × reference.py) |
| L7    | Perf regression                            | `scripts/bench_local.sh` → `bench_compare.py`       |

Key invariants the tests protect, in one list:

- Diagnostics always carry `(file, line, col)` — asserted by
  `cli_parse_bad_exits_nonzero` + golden `expected.diag.txt`.
- `--disable-pass=<name>` actually removes the pass's visible effect
  (pass L4 contract).
- `schedule-cuda` emits the selected variant in `emit-lir` output
  (so we can audit which kernel will be called without running it).
- CPU codegen binary output == LIR interpreter output exactly.
- CUDA codegen output matches CPU within `atol=1e-4, rtol=1e-3`.

## 4. CI vs local

- **Local**: 32 tests (CPU + CUDA). This is the full gate.
- **GHA**: ~20 tests (CPU path only — runners have no GPU). Selected
  by `.github/workflows/ci.yml`. Verified locally by moving `.venv`
  aside and reconfiguring from scratch (see the "CI equivalent" block
  in PLAN.md §W11 验收命令).

## 5. Extension points

If you want to add a new builtin op:

1. Add a member to `OpKind` in `hir/ops.h`, a `toString` case, and wire
   `builtinKindFromName`.
2. Extend `hir/verifier.cpp` with shape rules and a diagnostic for each
   failure mode.
3. Extend `hir/lowering.cpp` to produce it from the AST.
4. Add LIR lowering + interpreter support.
5. Add CPU adapter binding, then CUDA adapter binding.
6. Add L3 adapter tests (positive + at least one layout/shape neg case).
7. Add a golden example in `examples/` + an `emit-hir`/`emit-lir`/
   `run-lir` golden snapshot.
8. Extend the L6 e2e suite if the op changes model-level numerics.

If you want to add a new pass:

1. Drop a new file in `src/passes/`, export a free function `runFoo`.
2. Register it in `pass_manager.cpp` in the right pipeline slot (and
   only under `O1` by default unless it's a safety pass).
3. Add L4 test covering the four required assertions.

## 6. Known deviations from the paper plan

These are documented so future readers don't treat them as bugs:

- `View` / `Permute` are in the grammar but currently rejected by the
  verifier. The slot is held; payloads would follow the extension
  protocol above.
- Baseline scope is **3 rows**, not 18. See
  [`benchmarks/baseline.md`](benchmarks/baseline.md) for the empirical
  evidence that shrank this.
- `const-fold` is a named pipeline slot; it does not currently fold
  anything because no op has a folding rule yet. This is deliberate —
  keeping the slot named preserves the L4 test contract.
