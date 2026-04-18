#!/usr/bin/env python3
"""Reference-value generator for the W4 run-lir interpreter.

The interpreter fills each tensor parameter with the deterministic rule
`value[elem] = buf_idx * 0.5 + elem * 0.1`, evaluated per-function starting
from buf_idx=0 for the first tensor parameter.

This script mirrors that rule in numpy so humans can cross-check the C++
interpreter's printed output against what a standard tensor library would
produce, per the "interpreter vs numpy" requirement in PLAN.md §W4.

Usage:
    python scripts/compare_numpy.py matmul 2 2 2
    python scripts/compare_numpy.py add 1 4
    python scripts/compare_numpy.py softmax 1 4
    python scripts/compare_numpy.py rmsnorm 1 4
    python scripts/compare_numpy.py matmul 1 3 1        # odd shape

Outputs the reference result with the same formatting the interpreter uses
so the two dumps can be diffed visually.
"""

from __future__ import annotations
import sys

import numpy as np


def det_fill(buf_idx: int, shape: tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape))
    out = np.array([buf_idx * 0.5 + i * 0.1 for i in range(n)], dtype=np.float32)
    return out.reshape(shape)


def rmsnorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sq = (x * x).mean(axis=-1, keepdims=True)
    rms = np.sqrt(sq + eps)
    return x / rms


def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def dump(name: str, t: np.ndarray) -> None:
    shape = list(t.shape)
    print(f"  {name} shape={shape}:")
    inner = shape[-1] if shape else 1
    outer = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    flat = t.reshape(outer, inner)
    for row in flat:
        print("   " + "".join(f" {float(v):8.4f}" for v in row))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    op = argv[1]
    dims = [int(x) for x in argv[2:]]

    if op == "matmul":
        if len(dims) != 3:
            print("matmul needs M K N"); return 2
        M, K, N = dims
        A = det_fill(0, (M, K))
        B = det_fill(1, (K, N))
        C = A @ B
        dump("A", A); dump("B", B); dump("C", C)
    elif op == "add":
        if len(dims) < 1:
            print("add needs shape"); return 2
        A = det_fill(0, tuple(dims))
        B = det_fill(1, tuple(dims))
        C = A + B
        dump("A", A); dump("B", B); dump("C", C)
    elif op == "softmax":
        if len(dims) < 1:
            print("softmax needs shape"); return 2
        X = det_fill(0, tuple(dims))
        Y = softmax(X)
        dump("X", X); dump("Y", Y)
    elif op == "rmsnorm":
        if len(dims) < 1:
            print("rmsnorm needs shape"); return 2
        X = det_fill(0, tuple(dims))
        Y = rmsnorm(X)
        dump("X", X); dump("Y", Y)
    else:
        print(f"unknown op: {op}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
