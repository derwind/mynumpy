from __future__ import annotations

from ..core import ndarray, zeros


def eye(N: int, M: int | None = None, dtype=float) -> ndarray:
    if M is None:
        M = N
    arr = zeros((N, M), dtype=dtype)
    one = dtype(1)
    for i in range(min(N, M)):
        arr.data[i][i] = one
    return arr


def diag(v: ndarray):
    """
    Extract a diagonal or construct a diagonal array.
    """

    if isinstance(v, list) and not isinstance(v[0], list):
        n = len(v)
        a = zeros((n, n), type(v[0]))
        for i in range(n):
            a[i, i] = v[i]
        return a

    if len(v.shape) == 1:
        n = v.shape[0]
        a = zeros((n, n), v.dtype)
        for i in range(n):
            a[i, i] = v.data[i]
        return a
    elif len(v.shape) == 2:
        n = v.shape[0]
        return ndarray((n,), v.dtype, [v.data[i, i] for i in range(n)])
    else:
        raise ValueError("v must be 1- or 2-dimensional")
