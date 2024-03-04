from __future__ import annotations

import copy as copy_
from typing import List

from ..dtypes import Numbers
from .ndarray import _guess_dtype, calc_shape, ndarray


def array(data: List[Numbers], dtype=None, *, copy: bool = True) -> ndarray:
    if copy:
        data = copy_.deepcopy(data)
    if dtype is None:
        dtype = _guess_dtype(data)
    return ndarray(calc_shape(data), dtype, data)
