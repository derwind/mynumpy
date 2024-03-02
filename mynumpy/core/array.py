from __future__ import annotations

import copy as copy_
from typing import List
from .ndarray import ndarray, calc_shape, is_number
from ..dtypes import Numbers


def array(data: List[Numbers], dtype=None, *, copy: bool = True) -> ndarray:
    import mynumpy as mynp

    if copy:
        data = copy_.deepcopy(data)
    if dtype is None:
        dtype = _guess_dtype(data)
    return mynp.ndarray(calc_shape(data), dtype, data)


def _guess_dtype(data: Numbers | List[Numbers]) -> type:
    if is_number(data):
        return type(data)

    if isinstance(data[0], list):
        return _guess_dtype(data[0])
    return type(data[0])
