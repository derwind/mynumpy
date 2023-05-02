import copy
from typing import List, Tuple, Dict, Union, Optional, Any
from ..dtypes import Numbers


class ndarray:
    def __init__(self, data: Union[Numbers, List[Numbers]]):
        self.data = data
        self._shape = calc_shape(self.data)
        self._size = calc_size(self._shape)

    def __str__(self) -> str:
        return f'ndarray({str(self.data)})'

    def __repr__(self) -> str:
        return f'ndarray({str(self.data)})'

    def __eq__(self, other: 'ndarray') -> bool:
        if not isinstance(other, ndarray) and not is_number(other):
            return False
        if is_number(other):
            return self.data == other
        return self.data == other.data

    def __ne__(self, other: 'ndarray') -> bool:
        if not isinstance(other, ndarray) and not is_number(other):
            return True
        if is_number(other):
            return self.data != other
        return self.data != other.data

    def _prepare_operations(self, other: Union[Numbers, 'ndarray']) -> Tuple[List[int], List[int]]:
        a = self.flatten().data
        if is_number(other):
            b = [other] * self.size
        elif isinstance(other, ndarray) and is_number(other.data):
            b = [other.data] * self.size
        elif isinstance(other, list):
            other_shape = calc_shape(other)
            if not binary_operable(self.shape, other_shape):
                raise ValueError(f'operands could not be broadcast together with shapes {self.shape} {other_shape}')
            b = ndarray(list).flatten().data
        else:
            other_shape = other.shape
            if not binary_operable(self.shape, other.shape):
                raise ValueError(f'operands could not be broadcast together with shapes {self.shape} {other_shape}')
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
            a = a.reshape((1, a.shape[0]))
            assert len(a.shape) == 2
            squeeze_count += 1
        if len(b.shape) == 1:
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

        m = ndarray(placeholder).reshape(n_row, n_col)
        if need_transpose:
            m = m.T
        while squeeze_count > 0:
            m = ndarray(m.data[0])
            squeeze_count -= 1

        return m

    def __truediv__(self, other: Union[Numbers, 'ndarray']) -> 'ndarray':
        a, b = self._prepare_operations(other)

        return ndarray([x / y for x, y in zip(a, b)]).reshape(self.shape)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def size(self) -> int:
        return self._size

    def _transpose(self) -> List[Numbers]:
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

        data = self._flatten()
        for d in reversed(shape[1:]):
            if d != len(data):
                data = list(split_list(data, d))
            if shape[0] == 1:
                data = [data]

        return data

    def reshape(self, shape, *args) -> 'ndarray':
        return ndarray(self._reshape(shape, *args))

    def item(self) -> Numbers:
        if is_number(self.data):
            return self.data
        elif self.size == 1:
            data = self.data
            for _ in range(len(self.shape)):
                data = data[0]
            return data

        raise ValueError('can only convert an array of size 1 to a Python scalar')


def calc_shape(a: Union[Numbers, List[int]], dims: Optional[List[int]] = None) -> List[int]:
    if dims is None:
        dims = []
    if is_number(a):
        return tuple(dims)
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
    if is_number(a):
        return ndarray(0)
    elif isinstance(a, ndarray):
        a = a.data
    shape = calc_shape(a)

    return zeros(shape)


def _ones(shape: Union[int, List[int], Tuple[int]]) -> List[int]:
    return _numbers(shape, 1)


def ones(shape) -> 'ndarray':
    return ndarray(_ones(shape))


def ones_like(a) -> 'ndarray':
    if is_number(a):
        return ndarray(1)
    elif isinstance(a, ndarray):
        a = a.data
    shape = calc_shape(a)

    return ones(shape)


def is_number(n: Any):
    return isinstance(n, int) or isinstance(n, float) or isinstance(n, complex)


def binary_operable(shape_a: Union[int, List[int], Tuple[int]], shape_b: Union[int, List[int], Tuple[int]]) -> bool:
    if is_number(shape_a):
        return True

    if is_number(shape_b):
        return True

    shape_a = list(shape_a)
    shape_b = list(shape_b)

    if shape_a == shape_b:
        return True

    # operable if broadcast
    # XXX: very simple version
    return (shape_a[1:] == shape_b[1:]) and (shape_a[0] == 1 or shape_b[0] == 1)


def broadcast(a, shape: Union[List[int], Tuple[int]]) -> 'ndarray':
    # XXX: very simple version
    if binary_operable(a.shape, shape) or a.shape[0] != 1:
        return a

    n = shape[0]

    return ndarray([copy.deepcopy(a.data) for _ in range(n)])


def einsum(subscripts: str, *operands: List[ndarray]) -> ndarray:
    subscripts = subscripts.replace(' ', '')

    from_indices, to_index = subscripts.split('->')
    if len(from_indices.split(',')) != len(operands):
        raise ValueError('more operands provided to einstein sum function than specified in the subscripts string')

    index_list = [[idx for idx in index] for index in from_indices.split(',')]
    to_index = [idx for idx in to_index]

    for i, (op, index) in enumerate(zip(operands, index_list)):
        if len(op.shape) > len(index):
            raise ValueError('operand has more dimensions than subscripts given in einstein sum')

        if len(op.shape) < len(index):
            raise ValueError(f'einstein sum subscripts string contains too many subscripts for operand {i}')

    if len(operands) != 2:
        raise ValueError(f'operands whose length != 2 are currently not supported')

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

    def fill_placeholder(target: List[int], index: List[str], index_kv: Optional[Dict[str, int]] = None) -> List[int]:
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
    def calc_value(a_1: ndarray, a_2: ndarray, index_1: Tuple[str, ...], index_2: Tuple[str, ...], index_kv: Dict[str, int]):
        combinations_kv = []
        calc_combinations(list(a_1.shape), list(a_2.shape), index_1, index_2, index_kv, combinations_kv)

        v = 0
        for idx_kv in combinations_kv:
            v_1 = get_value(a_1.data, index_1, idx_kv)
            v_2 = get_value(a_2.data, index_2, idx_kv)
            v += v_1 * v_2

        return v

    def calc_combinations(
        shape_1: List[int], shape_2: List[int], index_1: List[str], index_2: List[str], index_kv: Dict[str, int], out_combs: List[Dict[str, int]]
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

    def get_value(target: List[Numbers], index: List[Numbers], index_kv: Dict[str, int]):
        if isinstance(target, list):
            idx, index = index[0], index[1:]
            target = target[index_kv[idx]]
            return get_value(target, index, index_kv)
        return target

    placeholder = fill_placeholder(placeholder, to_index)

    return ndarray(placeholder)
