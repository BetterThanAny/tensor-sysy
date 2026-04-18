"""Shared helpers for tensor-sysy e2e tests.

Provides:
  REPO_ROOT / BUILD_DIR / TSC — filesystem anchors.
  run_backend(backend, tsy_file) — run tsc subprocess, return stdout.
  parse_run_lir_output(stdout, buf_name) — pull a named tensor out of
      the printed run-lir report as a numpy array.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
TSC = BUILD_DIR / "tsc"


def run_backend(backend: str, tsy_file: Path) -> str:
    if not TSC.exists():
        pytest.skip(f"tsc not built at {TSC}; run cmake --build build first")
    cmd = [str(TSC), "run-lir", f"--backend={backend}", str(tsy_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if "requires tsy_runtime_cuda" in proc.stderr:
            pytest.skip(f"backend={backend} not built (no CUDA runtime)")
        raise RuntimeError(
            f"tsc run-lir --backend={backend} failed with rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def _parse_floats(tokens: list[str]) -> list[float]:
    out = []
    for tok in tokens:
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def parse_run_lir_output(stdout: str, buf_name: str) -> np.ndarray:
    """Extract buffer named `buf_name` from tsc run-lir stdout.

    Format (see src/lir/interpreter.cpp printRunResult):
        (local|input) <name> shape=[d0,d1,...]:
            <rows of floats>
    """
    header_pat = rf"(?:local|input)\s+{re.escape(buf_name)}\s+shape=\[([0-9,\s]+)\]:"
    m = re.search(header_pat, stdout)
    assert m, f"buffer {buf_name!r} not found in run-lir output"
    shape = tuple(int(x.strip()) for x in m.group(1).split(",") if x.strip())
    expected = int(np.prod(shape))

    tail = stdout[m.end():]
    # Body ends at next "local"/"input"/"function:" header or EOF.
    stop = re.search(r"\n\s*(?:input |local |function:)", tail)
    body = tail[: stop.start()] if stop else tail
    nums = _parse_floats(body.split())
    assert len(nums) >= expected, (
        f"buffer {buf_name!r}: parsed {len(nums)} floats, expected {expected}"
    )
    return np.asarray(nums[:expected], dtype=np.float32).reshape(shape)
