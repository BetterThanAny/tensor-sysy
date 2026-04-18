"""Numpy reference forward pass for examples/transformer_block.tsy.

Fill rule mirrors src/lir/interpreter.cpp:
    value[elem_idx] = buf_idx * 0.5 + elem_idx * 0.1
applied per parameter before reshape.

Param order matches the transformer_block(...) signature:
    0: x      [S, D]
    1: Wq     [D, D]
    2: Wk     [D, D]
    3: Wv     [D, D]
    4: Wo     [D, D]
    5: W1     [D, F]
    6: W2     [F, D]
"""
from __future__ import annotations

import numpy as np

S, D, F = 4, 8, 16


def det_fill(buf_idx: int, shape: tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape))
    flat = np.array(
        [buf_idx * 0.5 + i * 0.1 for i in range(n)],
        dtype=np.float32,
    )
    return flat.reshape(shape)


def rmsnorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sq = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(sq + eps)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax_lastdim(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def forward() -> np.ndarray:
    x = det_fill(0, (S, D))
    Wq = det_fill(1, (D, D))
    Wk = det_fill(2, (D, D))
    Wv = det_fill(3, (D, D))
    Wo = det_fill(4, (D, D))
    W1 = det_fill(5, (D, F))
    W2 = det_fill(6, (F, D))

    x_n = rmsnorm(x)
    Q = x_n @ Wq
    K = x_n @ Wk
    V = x_n @ Wv
    Kt = K.T  # (D, S)
    scores = Q @ Kt  # (S, S) — no sqrt(d) scaling, matches the fixture
    attn = softmax_lastdim(scores)
    ctx = attn @ V  # (S, D)
    a_out = ctx @ Wo
    x1 = x + a_out

    x1_n = rmsnorm(x1)
    h = x1_n @ W1  # (S, F)
    h1 = relu(h)
    f_out = h1 @ W2  # (S, D)
    out = x1 + f_out
    return out
