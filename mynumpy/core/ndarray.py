import copy


class ndarray:
    def __init__(self, data):
        self.data = copy.deepcopy(data)

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

        return count_dims(self.data, [])

    @property
    def size(self):
        size_ = 1
        for d in self.shape:
            size_ *= d
        return size_

    @property
    def T(self):
        ...

    def flatten(self):
        def walk(data, list_):
            if not isinstance(data, list):
                list_.append(data)
                return list_
            for subdata in data:
                list_ = walk(subdata, list_)
            return list_

        return walk(self.data, [])

    def reshape(self, shape):
        ...
