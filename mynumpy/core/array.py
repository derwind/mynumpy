import copy
from typing import List
from .ndarray import ndarray
from ..dtypes import Numbers


def array(data: List[Numbers], copy_: bool = True) -> ndarray:
    import mynumpy as mynp

    if copy_:
        data = copy.deepcopy(data)
    return mynp.ndarray(data)
