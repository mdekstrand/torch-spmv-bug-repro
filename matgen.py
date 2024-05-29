from typing import Literal

import torch
import numpy as np
import scipy.sparse as sps

from hypothesis import assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

@st.composite
def coo_arrays(
    draw,
    shape=None,
    dtype=nph.floating_dtypes(endianness="=", sizes=[32, 64]),
    elements=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False, width=32),
) -> sps.coo_array:
    if shape is None:
        shape = st.tuples(st.integers(1, 100), st.integers(1, 100))

    if isinstance(shape, st.SearchStrategy):
        shape = draw(shape)

    if not isinstance(shape, tuple):
        shape = shape, shape
    rows, cols = shape
    if isinstance(rows, st.SearchStrategy):
        rows = draw(rows)
    if isinstance(cols, st.SearchStrategy):
        cols = draw(cols)

    mask = draw(nph.arrays(np.bool_, (rows, cols)))
    # at least one nonzero value
    assume(np.any(mask))
    nnz = int(np.sum(mask))

    ris, cis = np.nonzero(mask)

    vs = draw(
        nph.arrays(dtype, nnz, elements=elements),
    )

    return sps.coo_array((vs, (ris, cis)), shape=(rows, cols))

def torch_sparse_from_scipy(
    M: sps.coo_array, layout: Literal["csr", "coo", "csc"] = "coo"
) -> torch.Tensor:
    """
    Convert a SciPy :class:`sps.coo_array` into a torch sparse tensor.
    """
    ris = torch.from_numpy(M.row)
    cis = torch.from_numpy(M.col)
    vs = torch.from_numpy(M.data)
    indices = torch.stack([ris, cis])
    assert indices.shape == (2, M.nnz)
    T = torch.sparse_coo_tensor(indices, vs, size=M.shape)
    assert T.shape == M.shape

    match layout:
        case "csr":
            return T.to_sparse_csr()
        case "csc":
            return T.to_sparse_csc()
        case "coo":
            return T.coalesce()
        case _:
            raise ValueError(f"invalid layout {layout}")
