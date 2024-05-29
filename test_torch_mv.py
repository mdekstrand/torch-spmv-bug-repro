import numpy as np
import torch
import scipy.sparse as sps

from pytest import approx
from hypothesis import settings, given, HealthCheck, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

from matgen import torch_sparse_from_scipy, coo_arrays


@settings(deadline=1000, max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(st.data(), coo_arrays(dtype="f8", shape=(1000, 1000)), st.sampled_from(["coo", "csr"]))
def test_torch_spmv(data, M: sps.coo_array, layout):
    "Test to make sure Torch spmv is behaved"
    nr, nc = M.shape
    v = data.draw(
        nph.arrays(
            M.data.dtype,
            nc,
            elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False, width=32),
        )
    )
    assume(not np.any(np.isnan(v)))
    res = M @ v
    assert np.all(np.isfinite(res))

    TM = torch_sparse_from_scipy(M, layout)
    tv = torch.from_numpy(v)

    # quick make sure that dense works
    assert M.todense() @ v == approx(res, rel=1.0e-5, abs=1.0e-9)
    assert torch.mv(torch.from_numpy(M.todense()), tv).numpy() == approx(
        res, rel=1.0e-5, abs=1.0e-9
    )

    tres = torch.mv(TM, tv)
    # tres = tres.nan_to_num()

    assert tres.numpy() == approx(res, rel=1.0e-5, abs=1.0e-9)
