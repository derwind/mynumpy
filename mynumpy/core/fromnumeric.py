from __future__ import annotations

from typing import Sequence

from ..dtypes import Numbers
from .ndarray import is_number, ndarray


def prod(a: Sequence[Numbers], dtype: type | None = None):
    if is_number(a):
        return a

    if dtype is None:
        dtype = type(a[0])

    result = 1
    for n in a:
        result *= n
    return dtype(result)


def allclose(a: ndarray, b: ndarray, rtol=1.0e-5, atol=1.0e-8):
    if a.shape != b.shape:
        return False
    a = a.flatten()
    b = b.flatten()
    for v1, v2 in zip(a, b):
        if abs(v1 - v2) > atol + rtol * abs(v2):
            return False
    return True
