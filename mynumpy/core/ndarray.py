class ndarray():
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def __eq__(self, other):
        ...

    def  __ne__(self, other):
        ...

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

    @property
    def ndim(self):
        ...

    @property
    def shape(self):
        ...

    @property
    def size(self):
        ...

    @property
    def T(self):
        ...

    def reshape(self, shape):
        ...
