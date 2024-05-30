from itertools import product

import numpy as np
import torch

from pytest import approx, mark, skip
from hypothesis import settings, given, HealthCheck, note
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


@st.composite
def mat_vec_mul_problems(draw):
    # create a matrix
    dtype = draw(nph.floating_dtypes(endianness="=", sizes=[64]))
    width = np.finfo(dtype).bits
    matrix = draw(
        nph.arrays(
            dtype,
            st.tuples(st.integers(1, 1000), st.integers(1, 1000)),
            elements=st.floats(
                -1e6, 1e6, allow_nan=False, allow_infinity=False, width=width
            ),
        )
    )

    # select items to zero out
    nr, nc = matrix.shape
    mask = draw(nph.arrays(np.bool_, (nr, nc)))
    matrix[~mask] = 0

    # draw the vector
    vec = draw(
        nph.arrays(
            dtype,
            nc,
            elements=st.floats(
                -1e6, 1e6, allow_nan=False, allow_infinity=False, width=width
            ),
        )
    )
    return matrix, vec


@mark.parametrize(
    "layout,dtype", product(["coo", "csr", "csc"], [np.float32, np.float64])
)
@settings(
    deadline=1000, max_examples=1000, suppress_health_check=[HealthCheck.too_slow]
)
@given(st.data(), st.integers(1, 500), st.integers(1, 500))
def test_torch_spmv(layout, dtype, data, nrows, ncols):
    "Test to make sure Torch spmv is behaved"
    if layout == "csc":
        skip("csc not documented to work")

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
    tres = torch.mv(TS, tv)
    # and check the result
    assert tres.numpy() == approx(res, rel=rtol, abs=atol)
