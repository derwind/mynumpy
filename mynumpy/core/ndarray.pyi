from typing import List, Tuple, Union
from mynumpy.dtypes import Numbers

class ndarray:
    def __init__(data: List[Numbers]): ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __eq__(self, other: ndarray) -> bool: ...

    def  __ne__(self, other: ndarray) -> bool: ...

    def __add__(self, other: Union[Numbers, ndarray]) -> ndarray: ...

    def __sub__(self, other: Union[Numbers, ndarray]) -> ndarray: ...

    def __mul__(self, other: Union[Numbers, ndarray]) -> ndarray: ...

    def __matmul__(self, other: ndarray) -> ndarray: ...

    def __truediv__(self, other: Union[Numbers, ndarray]) -> ndarray: ...

    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> Tuple[int, ...]: ...

    @property
    def size(self) -> int: ...

    @property
    def T(self) -> ndarray: ...

    def reshape(self, shape: Union[List[int], Tuple[int]]) -> ndarray: ...