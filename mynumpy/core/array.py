from __future__ import annotations

import copy as copy_
from typing import List

from ..dtypes import Numbers
from .ndarray import calc_shape, is_number, ndarray, zeros, _guess_dtype


def array(data: List[Numbers], dtype=None, *, copy: bool = True) -> ndarray:
    if copy:
        data = copy_.deepcopy(data)
    if dtype is None:
        dtype = _guess_dtype(data)
    return ndarray(calc_shape(data), dtype, data)


def eye(N: int, M: int | None = None, dtype=float) -> ndarray:
    if M is None:
        M = N
    arr = zeros((N, M), dtype=dtype)
    one = dtype(1)
    for i in range(min(N, M)):
        arr.data[i, i] = one
    return arr
