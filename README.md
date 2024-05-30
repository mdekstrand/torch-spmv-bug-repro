# Torch Bug Repro

This repository is a reproducer for a bug in PyTorch's sparse matrix-vector
multiply support (`torch.mv` applied to a CSR tensor).

It uses Hypothesis to generate sparse matrices, and then compares the results of
`torch.mv` with SciPy's matrix-vector multiply.  It also checks the results of
dense matrix-vector multiply with both Torch and Numpy to confirm that it is
actually Torch's sparse matrix-vector that is incorrect.

The test is run with both COO and CSR sparse matrices, to isolate the problem to
one or the other.
