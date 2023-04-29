import copy


class ndarray:
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __eq__(self, other):
        if not isinstance(other, ndarray):
            return False
        return self.data == other.data

    def __ne__(self, other):
        if not isinstance(other, ndarray):
            return True
        return self.data != other.data

    def __add__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __matmul__(self, other):
        ...

    def __truediv__(self, other):
        ...

    def __len__(self):
        return len(self.data)

    @property
    def ndim(self):
        def count_dim(data, count):
            if not isinstance(data, list):
                return count
            return count_dim(data[0], count + 1)

        return count_dim(self.data, 0)

    @property
    def shape(self):
        def count_dims(data, dims):
            if not isinstance(data, list):
                return dims
            dims.append(len(data))
            return count_dims(data[0], dims)

        dims = count_dims(self.data, [])
        if len(dims) <= 1:
            return (dims[0],)
        return tuple(dims)

    @property
    def size(self):
        return self.calc_size(self.shape)

    @property
    def T(self):
        ...

    def _flatten(self):
        def walk(data, list_):
            if not isinstance(data, list):
                list_.append(data)
                return list_
            for subdata in data:
                list_ = walk(subdata, list_)
            return list_

        return walk(self.data, [])

    def flatten(self):
        return ndarray(self._flatten())

    def _reshape(self, shape, *args):
        def split_list(l, n):
            for idx in range(0, len(l), n):
                yield l[idx : idx + n]

        if len(args) > 0:
            shape = [shape] + list(args)

        shape = list(shape)
        if shape[0] != -1:
            if self.size != self.calc_size(shape):
                raise ValueError(f'cannot reshape array of size {self.size} into shape {tuple(shape)}')
        elif shape[0] == -1:
            subsize = self.calc_size(shape[1:])
            if self.size % subsize != 0:
                raise ValueError(f'cannot reshape array of size {self.size} into shape {tuple(shape)}')
            shape[0] = self.size // subsize

        if self.size % self.calc_size(shape) != 0:
            raise ValueError(f'cannot reshape array of size {self.size} into shape {tuple(shape)}')

        # confirmed valid shape

        data = self._flatten()
        for d in reversed(shape[1:]):
            if d != len(data):
                data = list(split_list(data, d))

        return data

    def reshape(self, shape, *args):
        return ndarray(self._reshape(shape, *args))

    @staticmethod
    def calc_size(shape):
        size_ = 1
        for d in shape:
            size_ *= d
        return size_
