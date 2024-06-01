"""
Exactly the same test as test_torch_mv, except with mm to demonstrate
the problem lines in MV.
"""

from itertools import product

import numpy as np
import torch

from pytest import approx, mark, skip
from hypothesis import given, note
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


@mark.parametrize("layout,dtype", product(["coo", "csr"], [np.float32, np.float64]))
@given(st.data(), st.integers(1, 500), st.integers(1, 500))
def test_torch_spmm(layout, dtype, data, nrows, ncols):
    "Test to make sure Torch spmm is behaved"
    if dtype == np.float32:
        skip("float32 too noisy to reliably test")

    # draw the initial matrix
    M = data.draw(
        nph.arrays(
            dtype,
            (nrows, ncols),
            elements=st.floats(
                -1e4,
                1e4,
                allow_nan=False,
                allow_infinity=False,
                width=np.finfo(dtype).bits,
            ),
        )
    )
    # draw the vector
    v = data.draw(
        nph.arrays(
            dtype,
            ncols,
            elements=st.floats(
                -1e4,
                1e4,
                allow_nan=False,
                allow_infinity=False,
                width=np.finfo(dtype).bits,
            ),
        )
    )
    # zero out items in the matrix
    mask = data.draw(nph.arrays(np.bool_, (nrows, ncols)))
    M[~mask] = 0.0
    nnz = np.sum(M != 0.0)
    note("matrix {} x {} ({} nnz)".format(nrows, ncols, nnz))

    # make our tolerance depend on the data type we got
    if dtype == np.float64:
        rtol, atol = 1.0e-5, 1.0e-4
    elif dtype == np.float32:
        rtol, atol = 0.05, 1.0e-3
    else:
        raise TypeError(f"unexpected data type {dtype}")

    # multiply them (dense operation with NumPy) to get expected result
    res = M @ v
    # just make sure everything's finite, should always pass due to data limits
    assert np.all(np.isfinite(res))

    # convert to Torch dense tensor, to make sure we get the right result
    # (this should always pass, it isn't the bug)
    T = torch.from_numpy(M)
    tv = torch.from_numpy(v)
    assert torch.mv(T, tv).numpy() == approx(res, rel=rtol, abs=atol)

    # and now we do the sparse multiplication
    # first make the tensor sparse
    match layout:
        case "coo":
            TS = T.to_sparse_coo().coalesce()
        case "csr":
            TS = T.to_sparse_csr()
        case "csc":
            TS = T.to_sparse_csc()
        case _:
            raise ValueError(f"unknown layout {layout}")

    # then multiply
    tres = torch.mm(TS, tv.reshape(-1, 1)).reshape(-1)
    # and check the result
    assert tres.numpy() == approx(res, rel=rtol, abs=atol)
