from __future__ import annotations

import copy
import random
from typing import Dict, Any
from ..dtypes import Numbers


class ndarray:
    def __init__(self, shape, dtype: type = float, data: Numbers | list[Numbers] | None = None):
        self._size = calc_size(shape)
        self._shape = (self._size,)  # temporary shape

        # make flat data and convert to specified type
        if data is None:
            self.data = [dtype((random.random() - 0.5) * 2) for _ in range(self._size)]
        else:
            self.data = ndarray._flatten(data, dtype)

        # reshape to specified shape
        self.data = self._reshape(shape)
        self._dtype = dtype
        self._shape = shape  # overwrite shape with specified shape

    def __str__(self) -> str:
        return f'ndarray({str(self.data)})'

    def __repr__(self) -> str:
        return f'ndarray({str(self.data)})'

    def __eq__(self, other: ndarray) -> bool:
        if not isinstance(other, ndarray) and not is_number(other):
            return False
        if is_number(other):
            return self.data == other
        return self.data == other.data

    def __ne__(self, other: ndarray) -> bool:
        if not isinstance(other, ndarray) and not is_number(other):
            return True
        if is_number(other):
            return self.data != other
        return self.data != other.data

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.data[key]

        assert isinstance(key, tuple)

        if len(key) > len(self.shape):
            raise IndexError(f'too many indices for array: array is {len(self.shape)}-dimensional, but {len(key)} were indexed')

        shape = []
        indices = []
        for i in range(len(self.shape)):
            if i < len(key):
                subkey = key[i]
                if not isinstance(subkey, slice):
                    indices.append([subkey])
                    continue
                else:
                    start = subkey.start or 0
                    stop = subkey.stop or self.shape[i]
                    step = subkey.step or 1
                    indice = list(range(start, stop, step))
                    shape.append(len(indice))
                    indices.append(indice)
            else:
                shape.append(self.shape[i])
                indices.append(list(range(self.shape[i])))

        def walk(data: Numbers | list, indices: list[int], outputs: list[int]):
            if is_number(data):
                outputs.append(data)
                return

            for idx in indices[0]:
                walk(data[idx], indices[1:], outputs)

        outputs = []
        walk(self.data, indices, outputs)

        return ndarray(shape, self.dtype, outputs)

    def __setitem__(self, key, value):
        if isinstance(key, int) or isinstance(key, slice):
            self.data[key] = value
            return

        assert isinstance(key, tuple)

        if len(key) > len(self.shape):
            raise IndexError(f'too many indices for array: array is {len(self.shape)}-dimensional, but {len(key)} were indexed')

        shape = []
        indices = []
        for i in range(len(self.shape)):
            if i < len(key):
                subkey = key[i]
                if not isinstance(subkey, slice):
                    indices.append([subkey])
                    continue
                else:
                    start = subkey.start or 0
                    stop = subkey.stop or self.shape[i]
                    step = subkey.step or 1
                    indice = list(range(start, stop, step))
                    shape.append(len(indice))
                    indices.append(indice)
            else:
                shape.append(self.shape[i])
                indices.append(list(range(self.shape[i])))

        if not isinstance(value, ndarray):
            value = ndarray(calc_shape(value), self.dtype, value)
        value = broadcast(value, shape).flatten().data

        def walk(data: Numbers | list, indices: list[int], inputs: list[int]):
            if is_number(data[0]):  # list of Numbers
                for i in indices[0]:
                    v = inputs[0]
                    inputs[:] = inputs[1:]
                    data[i] = v
                return

            for idx in indices[0]:
                walk(data[idx], indices[1:], inputs)

        walk(self.data, indices, value)

    def _prepare_operations(self, other: Numbers | ndarray) -> tuple[list[int], list[int], list[int]]:
        new_shape = list(self.shape)

        if is_number(other):
            a = self.flatten().data
            b = [other] * self.size
        elif isinstance(other, ndarray) and is_number(other.data):  # scaler
            a = self.flatten().data
            b = [other.data] * self.size
        elif isinstance(other, list) or isinstance(other, ndarray):
            if isinstance(other, list):
                other = ndarray(calc_shape(other), self.dtype, other)
            is_operable, new_shape = binary_operable(self.shape, other.shape)
            if not is_operable:
                raise ValueError(f'operands could not be broadcast together with shapes {self.shape} {other.shape}')
            a = broadcast(self, new_shape)
            b = broadcast(other, new_shape)
            a = a.flatten().data
            b = b.flatten().data
        else:
            raise TypeError('ufunc did not contain a loop with signature matching types')

        return a, b, new_shape

    def __add__(self, other: Numbers | ndarray) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x + y for x, y in zip(a, b)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __radd__(self, other: Numbers) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x + y for x, y in zip(b, a)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __sub__(self, other: Numbers | ndarray) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x - y for x, y in zip(a, b)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __rsub__(self, other: Numbers) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x - y for x, y in zip(b, a)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __mul__(self, other: Numbers | ndarray) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x * y for x, y in zip(a, b)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __rmul__(self, other: Numbers) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x * y for x, y in zip(b, a)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __matmul__(self, other: ndarray) -> ndarray:
        if len(self.shape) < 1:
            raise ValueError(
                f'matmul: Input operand 0 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)'
            )
        if len(other.shape) < 1:
            raise ValueError(
                f'matmul: Input operand 1 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)'
            )

        if len(self.shape) != 1 and len(self.shape) != 2:
            raise ValueError(f'matmul: Input operand 0 is neither a vector nor a matrix and not supported')
        if len(other.shape) != 1 and len(other.shape) != 2:
            raise ValueError(f'matmul: Input operand 1 is neither a vector nor a matrix and not supported')

        a = self
        b = other
        squeeze_count = 0
        need_transpose = False
        if len(a.shape) == 1:
            # to a row vector of the matrix-form
            a = a.reshape((1, a.shape[0]))
            assert len(a.shape) == 2
            squeeze_count += 1
        if len(b.shape) == 1:
            # to a col vector
            b = b.reshape((b.shape[0], 1))
            assert len(b.shape) == 2
            need_transpose = True
            squeeze_count += 1

        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f'matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size {b.shape[0]} is different from {a.shape[1]})'
            )

        n_row = a.shape[0]
        n_col = b.shape[1]
        placeholder = _zeros((n_row, n_col))

        for r in range(n_row):
            for c in range(n_col):
                for i in range(a.shape[1]):
                    placeholder[r][c] += a.data[r][i] * b.data[i][c]

        m = ndarray(calc_shape(placeholder), self.dtype, placeholder).reshape(n_row, n_col)
        if need_transpose:
            m = m.T
        while squeeze_count > 0:
            m = ndarray(calc_shape(m.data[0]), self.dtype, m.data[0])
            squeeze_count -= 1

        return m

    def __truediv__(self, other: Numbers | ndarray) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x / y for x, y in zip(a, b)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __rtruediv__(self, other: Numbers) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x / y for x, y in zip(b, a)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __floordiv__(self, other: Numbers | ndarray) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x // y for x, y in zip(a, b)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __rfloordiv__(self, other: Numbers) -> ndarray:
        a, b, new_shape = self._prepare_operations(other)
        data = [x // y for x, y in zip(b, a)]
        return ndarray(calc_shape(data), type(data[0]), data).reshape(new_shape)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> tuple[int]:
        return self._shape

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def size(self) -> int:
        return self._size

    def _transpose(self) -> Numbers | list[Numbers]:
        if is_number(self.data):
            return self.data

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

        flat_data = ndarray._flatten(self.data)

        for d, index in zip(flat_data, indices):
            target = placeholder
            for idx in index[:-1]:
                target = target[idx]
            target[index[-1]] = d

        return placeholder

    @property
    def T(self) -> ndarray:
        data = self._transpose()
        return ndarray(calc_shape(data), self.dtype, data)

    @classmethod
    def _flatten(cls, data: Numbers | list[Numbers], dtype: type | None = None) -> list[Numbers]:
        def walk(data, list_, dtype=None):
            if not isinstance(data, list):
                if dtype is not None:
                    data = dtype(data)
                list_.append(data)
                return list_
            for subdata in data:
                list_ = walk(subdata, list_, dtype)
            return list_

        return walk(data, [], dtype)

    def flatten(self) -> ndarray:
        data = ndarray._flatten(self.data)
        return ndarray(calc_shape(data), self.dtype, data)

    def _reshape(self, shape, *args) -> list[Numbers]:
        def split_list(l, n):
            for idx in range(0, len(l), n):
                yield l[idx : idx + n]

        if len(args) > 0:
            shape = [shape] + list(args)
        elif isinstance(shape, int):
            shape = [shape]

        shape = list(shape)

        if not shape and self.size == 1:
            return self.item()

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

        data = ndarray._flatten(self.data)
        for dim in reversed(shape[1:]):
            data = list(split_list(data, dim))
        # shape[0]'s dimesion should be automatically sufficed

        assert calc_shape(data) == tuple(shape), f'{calc_shape(data)} != {tuple(shape)}'

        return data

    def reshape(self, shape, *args) -> ndarray:
        data = self._reshape(shape, *args)
        return ndarray(calc_shape(data), self.dtype, data)

    def item(self) -> Numbers:
        if is_number(self.data):
            return self.data
        elif self.size == 1:
            data = self.data
            for _ in range(len(self.shape)):
                data = data[0]
            return data

        raise ValueError('can only convert an array of size 1 to a Python scalar')


def calc_shape(a: Numbers | list[int], dims: list[int] | None = None) -> list[int]:
    if dims is None:
        dims = []
    if is_number(a):
        return tuple(dims)
    dims.append(len(a))
    return calc_shape(a[0], dims)


def calc_size(shape: int | list[int] | tuple[int], *args) -> int:
    if len(args) > 0:
        shape = [shape] + list(args)
    elif isinstance(shape, int):
        shape = [shape]

    size = 1
    for d in shape:
        size *= d
    return size


def _numbers(shape: int | list[int] | tuple[int], n: Numbers) -> list[Numbers]:
    if isinstance(shape, int):
        shape = [shape]
    data = [n] * calc_size(shape)
    return ndarray(shape, type(n), data).data


def _zeros(shape: int | list[int] | tuple[int]) -> list[int]:
    return _numbers(shape, 0)


def zeros(shape, dtype: type = float) -> ndarray:
    return ndarray(shape, dtype, _zeros(shape))


def zeros_like(a) -> ndarray:
    if is_number(a):
        return ndarray((), type(a), 0)
    elif isinstance(a, ndarray):
        a = a.data
    shape = calc_shape(a)

    return zeros(shape)


def _ones(shape: int | list[int] | tuple[int]) -> list[int]:
    return _numbers(shape, 1)


def ones(shape, dtype: type = float) -> ndarray:
    return ndarray(shape, dtype, _ones(shape))


def ones_like(a) -> ndarray:
    if is_number(a):
        return ndarray((), type(a), 1)
    elif isinstance(a, ndarray):
        a = a.data
    shape = calc_shape(a)

    return ones(shape)


def is_number(n: Any):
    return isinstance(n, int) or isinstance(n, float) or isinstance(n, complex)


def binary_operable(shape_a: list[int] | tuple[int], shape_b: list[int] | tuple[int]) -> tuple[bool, list[int]]:
    shape_a = list(shape_a)
    shape_b = list(shape_b)

    if shape_a == shape_b:
        return True, shape_a

    if len(shape_a) != len(shape_b):
        if len(shape_a) > len(shape_b):
            shape_b = [1] * (len(shape_a) - len(shape_b)) + shape_b
        else:
            shape_a = [1] * (len(shape_b) - len(shape_a)) + shape_a

    # now shape_a and shape_b have shapes with same length

    new_shape = []
    for dim1, dim2 in zip(shape_a, shape_b):
        if dim1 == 1:
            new_shape.append(dim2)
        elif dim2 == 1:
            new_shape.append(dim1)
        else:
            return False, []

    # operable if broadcast
    return True, new_shape


def is_broadcastable(a: ndarray, shape: list[int] | tuple[int]) -> bool:
    shape_a = list(a.shape)
    shape = list(shape)

    if len(shape_a) > len(shape):
        return False
    elif len(shape_a) < len(shape):
        shape_a = [1] * (len(shape) - len(shape_a)) + shape_a

    for dim1, dim2 in zip(shape_a, shape):
        if dim1 != dim2 and dim1 != 1:
            return False

    return True


def broadcast(a: ndarray, shape: list[int] | tuple[int]) -> ndarray:
    if not is_broadcastable(a, shape):
        raise ValueError(f'could not broadcast input array from shape {a.shape} into shape {tuple(shape)}')

    shape = list(shape)

    if a.shape == tuple(shape):
        return a

    if len(a.shape) < len(shape):
        data = copy.deepcopy(a.data)
        for _ in range(len(shape) - len(a.shape)):
            data = [data]
        a = ndarray(calc_shape(data), a.dtype, data)

    data = copy.deepcopy(a.data)

    def walk(data, shape):
        if not isinstance(data[0], list):
            if len(data) == 1 and shape[0] > 1:
                v = data[0]
                for _ in range(shape[0] - 1):
                    data.append(v)
            return

        for subdata in data:
            walk(subdata, shape[1:])
        if len(data) == 1 and shape[0] > 1:
            v = data[0]
            for _ in range(shape[0] - 1):
                data.append(copy.deepcopy(v))

    walk(data, shape)

    a = ndarray(calc_shape(data), a.dtype, data)
    assert a.shape == tuple(shape)

    return a


def einsum(subscripts: str, *operands: list[ndarray]) -> ndarray:
    subscripts = subscripts.replace(' ', '')

    from_indices, to_index = subscripts.split('->')
    if len(from_indices.split(',')) != len(operands):
        raise ValueError('more operands provided to einstein sum function than specified in the subscripts string')

    if len(operands) == 1:
        # XXX: ad-hoc implementation
        op = operands[0]
        operands = [op, ones_like(op)]
        from_indices = f'{from_indices},{from_indices}'

    index_list = [[idx for idx in index] for index in from_indices.split(',')]
    to_index = [idx for idx in to_index]

    for i, (op, index) in enumerate(zip(operands, index_list)):
        if len(op.shape) > len(index):
            raise ValueError('operand has more dimensions than subscripts given in einstein sum')

        if len(op.shape) < len(index):
            raise ValueError(f'einstein sum subscripts string contains too many subscripts for operand {i}')

    if len(operands) > 2:
        raise ValueError(f'operands whose length > 2 are currently not supported')

    a, b = operands
    index_a, index_b = index_list

    # index char -> loc, e.g. {'i': 0, 'j': 1, 'k': 2, 'l': 3} for 'ijkl'
    i2l_a = {index: index_a.index(index) for index in index_a}
    i2l_b = {index: index_b.index(index) for index in index_b}

    # determin output tensor's shape

    out_shape = []
    # index char -> dim, e.g. {'i': 3, 'j': 4}
    i2d = {}
    for idx in to_index:
        if idx in i2l_a:
            dim = a.shape[i2l_a[idx]]
            out_shape.append(dim)
            i2d[idx] = dim
            continue
        if idx in i2l_b:
            dim = b.shape[i2l_b[idx]]
            out_shape.append(dim)
            i2d[idx] = dim
            continue
        raise ValueError(f"einstein sum subscripts string included output subscript '{idx}' which never appeared in an input")

    # Preprocess finished. Main process begins

    placeholder = zeros(out_shape).data

    def fill_placeholder(target: list[int], index: list[str], index_kv: Dict[str, int] | None = None) -> list[int]:
        if index_kv is None:
            index_kv = {}

        if not index:
            # return scaler value
            return calc_value(a, b, index_a, index_b, index_kv)

        idx, index = index[0], index[1:]  # index chars

        for i in range(i2d[idx]):
            index_kv_ = index_kv.copy()
            index_kv_[idx] = i
            if isinstance(target[i], list):
                fill_placeholder(target[i], index, index_kv_)
                continue

            target[i] = calc_value(a, b, index_a, index_b, index_kv_)

        return target

    # e.g. 'ijkl,jmln->ikm': sum_j sum_l sum_n A_{ijkl} B_{jmln}
    def calc_value(a_1: ndarray, a_2: ndarray, index_1: tuple[str, ...], index_2: tuple[str, ...], index_kv: Dict[str, int]):
        combinations_kv = []
        calc_combinations(list(a_1.shape), list(a_2.shape), index_1, index_2, index_kv, combinations_kv)

        v = 0
        for idx_kv in combinations_kv:
            v_1 = get_value(a_1.data, index_1, idx_kv)
            v_2 = get_value(a_2.data, index_2, idx_kv)
            v += v_1 * v_2

        return v

    def calc_combinations(
        shape_1: list[int], shape_2: list[int], index_1: list[str], index_2: list[str], index_kv: Dict[str, int], out_combs: list[Dict[str, int]]
    ):
        if index_1:
            idx1, index_1 = index_1[0], index_1[1:]
            dim1, shape_1 = shape_1[0], shape_1[1:]
            if idx1 in index_kv:
                calc_combinations(shape_1, shape_2, index_1, index_2, index_kv, out_combs)
                return
            else:
                for i in range(dim1):
                    index_kv_ = index_kv.copy()
                    index_kv_[idx1] = i
                    calc_combinations(shape_1, shape_2, index_1, index_2, index_kv_, out_combs)
                return

        if index_2:
            idx2, index_2 = index_2[0], index_2[1:]
            dim2, shape_2 = shape_2[0], shape_2[1:]
            if idx2 in index_kv:
                calc_combinations(shape_1, shape_2, index_1, index_2, index_kv, out_combs)
                return
            else:
                for i in range(dim2):
                    index_kv_ = index_kv.copy()
                    index_kv_[idx2] = i
                    calc_combinations(shape_1, shape_2, index_1, index_2, index_kv_, out_combs)
                return

        out_combs.append(index_kv)

    def get_value(target: list[Numbers], index: list[Numbers], index_kv: Dict[str, int]):
        if isinstance(target, list):
            idx, index = index[0], index[1:]
            target = target[index_kv[idx]]
            return get_value(target, index, index_kv)
        return target

    placeholder = fill_placeholder(placeholder, to_index)

    return ndarray(calc_shape(placeholder), operands[0].dtype, placeholder)
