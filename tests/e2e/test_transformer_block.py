"""W10 e2e: compare 3 backends against numpy reference."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import REPO_ROOT, parse_run_lir_output, run_backend
from .reference import forward as ref_forward


TSY = REPO_ROOT / "examples" / "transformer_block.tsy"


@pytest.mark.parametrize("backend", ["native", "cpu-adapter", "cuda-adapter"])
def test_transformer_block_matches_numpy(backend):
    stdout = run_backend(backend, TSY)
    actual = parse_run_lir_output(stdout, "out")
    expected = ref_forward()

    assert actual.shape == expected.shape, (
        f"shape mismatch: got {actual.shape}, expected {expected.shape}"
    )
    np.testing.assert_allclose(
        actual, expected,
        atol=1e-3, rtol=1e-2,
        err_msg=f"backend={backend} differs from numpy reference",
    )
