#!/usr/bin/env python3
"""Reference values for the W7 MLP, computed with PyTorch.

Intended for the "PyTorch 对拍" check in PLAN.md §W7. PyTorch is not a CI
dependency; this script exists so a human can sanity-check the codegen
binary's output against an outside-the-project implementation when
building a confidence baseline.

Usage:
    python scripts/compare_pytorch.py

The script uses the same deterministic fill the interpreter and adapter
use: `value[elem] = buf_idx * 0.5 + elem * 0.1`, counted across the
tensor parameters in source order (X=0, W1=1, b1=2, W2=3, b2=4).
"""

from __future__ import annotations

import sys


def det_fill(buf_idx: int, shape: tuple[int, ...]):
    import numpy as np
    n = int(np.prod(shape))
    data = [buf_idx * 0.5 + i * 0.1 for i in range(n)]
    return np.array(data, dtype=np.float32).reshape(shape)


def main() -> int:
    try:
        import numpy as np  # noqa: F401
        import torch
    except ImportError as e:
        print(f"missing dependency: {e}. Install torch + numpy to use this script.",
              file=sys.stderr)
        return 2

    B, D1, D2, D3 = 2, 3, 4, 2
    X  = torch.from_numpy(det_fill(0, (B, D1)))
    W1 = torch.from_numpy(det_fill(1, (D1, D2)))
    b1 = torch.from_numpy(det_fill(2, (B, D2)))
    W2 = torch.from_numpy(det_fill(3, (D2, D3)))
    b2 = torch.from_numpy(det_fill(4, (B, D3)))

    h1   = X @ W1
    h1b  = h1 + b1
    h1s  = torch.softmax(h1b, dim=-1)
    h2   = h1s @ W2
    h2b  = h2 + b2
    y    = torch.softmax(h2b, dim=-1)

    def show(name, t):
        rows = t.shape[0] if t.ndim > 1 else 1
        cols = t.shape[-1]
        flat = t.reshape(rows, cols)
        print(f"  {name} shape={list(t.shape)}:")
        for r in range(rows):
            print("   " + "".join(f" {float(flat[r, c]):8.4f}" for c in range(cols)))

    print("function: mlp_forward")
    for name, t in [("X", X), ("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
        show(name, t)
    for name, t in [("h1", h1), ("h1b", h1b), ("h1s", h1s),
                    ("h2", h2), ("h2b", h2b), ("y", y)]:
        show(name, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
