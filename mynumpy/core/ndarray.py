import copy
from typing import List, Tuple, Union, Optional, Any
from ..dtypes import Numbers


class ndarray:
    def __init__(self, data: List[Numbers]):
        self.data = data

    def __str__(self) -> str:
        return f'ndarray({str(self.data)})'

    def __repr__(self) -> str:
        return f'ndarray({str(self.data)})'

    def __eq__(self, other: 'ndarray') -> bool:
        if not isinstance(other, ndarray):
            return False
        return self.data == other.data

    def __ne__(self, other: 'ndarray') -> bool:
        if not isinstance(other, ndarray):
            return True
        return self.data != other.data

    def _prepare_operations(self, other: Union[Numbers, 'ndarray']) -> Tuple[List[int], List[int]]:
        a = self.flatten().data
        if self.is_number(other):
            b = [other] * self.size
        elif isinstance(other, list):
            b = ndarray(list).flatten().data
        else:
            b = other.flatten().data

        return a, b

    def __add__(self, other: Union[Numbers, 'ndarray']) -> 'ndarray':
        a, b = self._prepare_operations(other)

        return ndarray([x + y for x, y in zip(a, b)]).reshape(self.shape)

    def __sub__(self, other: Union[Numbers, 'ndarray']) -> 'ndarray':
        a, b = self._prepare_operations(other)

        return ndarray([x - y for x, y in zip(a, b)]).reshape(self.shape)

    def __mul__(self, other: Union[Numbers, 'ndarray']) -> 'ndarray':
        a, b = self._prepare_operations(other)

        return ndarray([x * y for x, y in zip(a, b)]).reshape(self.shape)

    def __matmul__(self, other: 'ndarray') -> 'ndarray':
        ...

    def __truediv__(self, other: Union[Numbers, 'ndarray']) -> 'ndarray':
        a, b = self._prepare_operations(other)

        return ndarray([x / y for x, y in zip(a, b)]).reshape(self.shape)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def ndim(self) -> int:
        def count_dim(data, count):
            if not isinstance(data, list):
                return count
            return count_dim(data[0], count + 1)

        return count_dim(self.data, 0)

    @property
    def shape(self) -> Tuple[int]:
        dims = calc_shape(self.data)
        if len(dims) <= 1:
            return (dims[0],)
        return tuple(dims)

    @property
    def size(self) -> int:
        return calc_size(self.shape)

    def _transpose(self) -> List[Numbers]:
        def calc_target_indices(data, out_index_list):
            def walk(data, out_list, indices):
                if not isinstance(data[0], list):
                    for i in range(len(data)):
                        indices_ = indices[:]
                        indices_.append(i)
                        indices_ = list(reversed(indices_))
                        out_list.append(indices_)
                    return
                for i, subdata in enumerate(data):
                    indices_ = indices[:]
                    indices_.append(i)
                    walk(subdata, out_list, indices_)
                return

            walk(data, out_index_list, [])

        placeholder = _zeros(list(reversed(self.shape)))

        indices = []
        calc_target_indices(self.data, indices)

        flat_data = self._flatten()

        for d, index in zip(flat_data, indices):
            target = placeholder
            for idx in index[:-1]:
                target = target[idx]
            target[index[-1]] = d

        return placeholder

    @property
    def T(self) -> 'ndarray':
        return ndarray(self._transpose())

    def _flatten(self) -> List[Numbers]:
        def walk(data, list_):
            if not isinstance(data, list):
                list_.append(data)
                return list_
            for subdata in data:
                list_ = walk(subdata, list_)
            return list_

        return walk(self.data, [])

    def flatten(self) -> 'ndarray':
        return ndarray(self._flatten())

    def _reshape(self, shape, *args) -> List[Numbers]:
        def split_list(l, n):
            for idx in range(0, len(l), n):
                yield l[idx : idx + n]

        if len(args) > 0:
            shape = [shape] + list(args)
        elif isinstance(shape, int):
            shape = [shape]

        shape = list(shape)
        if shape[0] != -1:
            if self.size != calc_size(shape):
                raise ValueError(f'cannot reshape array of size {self.size} into shape {tuple(shape)}')
        elif shape[0] == -1:
            subsize = calc_size(shape[1:])
            if self.size % subsize != 0:
                raise ValueError(f'cannot reshape array of size {self.size} into shape {tuple(shape)}')
            shape[0] = self.size // subsize

        if self.size % calc_size(shape) != 0:
            raise ValueError(f'cannot reshape array of size {self.size} into shape {tuple(shape)}')

        # confirmed valid shape

        data = self._flatten()
        for d in reversed(shape[1:]):
            if d != len(data):
                data = list(split_list(data, d))
        if shape[0] == 1:
            data = [data]

        return data

    def reshape(self, shape, *args) -> 'ndarray':
        return ndarray(self._reshape(shape, *args))

    @staticmethod
    def is_number(n: Any):
        return isinstance(n, int) or isinstance(n, float) or isinstance(n, complex)


def calc_shape(a: Union[list, 'ndarray'], dims: Optional[List[int]] = None) -> List[int]:
    if isinstance(a, ndarray):
        return a.shape

    # list

    if dims is None:
        dims = []
    if not isinstance(a, list):
        return dims
    dims.append(len(a))
    return calc_shape(a[0], dims)


def calc_size(shape: Union[int, List[int], Tuple[int]], *args) -> int:
    if len(args) > 0:
        shape = [shape] + list(args)
    elif isinstance(shape, int):
        shape = [shape]

    size = 1
    for d in shape:
        size *= d
    return size


def _numbers(shape: Union[int, List[int], Tuple[int]], n: Numbers) -> List[int]:
    if isinstance(shape, int):
        shape = [shape]

    return ndarray([n] * calc_size(shape))._reshape(shape)


def _zeros(shape: Union[int, List[int], Tuple[int]]) -> List[int]:
    return _numbers(shape, 0)


def zeros(shape) -> 'ndarray':
    return ndarray(_zeros(shape))


def zeros_like(a) -> 'ndarray':
    if isinstance(a, ndarray):
        a = a.data
    shape = calc_shape(a)

    return zeros(shape)


def binary_operable(shape_a: Union[int, List[int], Tuple[int]], shape_b: Union[int, List[int], Tuple[int]]) -> bool:
    if not isinstance(shape_a, list) and not isinstance(shape_a, tuple):
        return True

    if not isinstance(shape_b, list) and not isinstance(shape_b, tuple):
        return True

    # XXX: very simple version
    return (shape_a[1:] == shape_b[1:]) and (shape_a[0] == 1 or shape_b[0] == 1)


def broadcast(a, shape: Union[List[int], Tuple[int]]) -> 'ndarray':
    # XXX: very simple version
    if binary_operable(a.shape, shape) or a.shape[0] != 1:
        return a

    n = shape[0]

    return ndarray([copy.deepcopy(a.data) for _ in range(n)])
