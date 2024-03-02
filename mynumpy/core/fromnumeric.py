from __future__ import annotations

from typing import Sequence

from ..dtypes import Numbers
from .ndarray import is_number


def prod(a: Sequence[Numbers], dtype: type | None = None):
    if is_number(a):
        return a

    if dtype is None:
        dtype = type(a[0])

    result = 1
    for n in a:
        result *= n
    return dtype(result)
