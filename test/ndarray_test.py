import unittest
import mynumpy as mynp

class TestNdArray(unittest.TestCase):
    def setUp(self):
        ...

    def tearDown(self):
        ...

    def test_create(self):
        data = 3
        a = mynp.array(data)
        self.assertEqual(data, a.data)

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual(data, a.data)

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual(data, a.data)

        data = [
            [
                [1, -2],
                [-3, 4]
            ],
            [
                [-5, 6],
                [7, -8]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual(data, a.data)

    def test_eq(self):
        data = 3
        a = mynp.array(data)
        b = mynp.array(data)
        self.assertTrue(a == 3)
        self.assertTrue(a == b)

        data = [1, 2, 3]
        a = mynp.array(data)
        b = mynp.array(data)

        self.assertTrue(a == b)

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)
        b = mynp.array(data)

        self.assertTrue(a == b)

        data = [
            [
                [1, -2],
                [-3, 4]
            ],
            [
                [-5, 6],
                [7, -8]
            ]
        ]
        a = mynp.array(data)
        b = mynp.array(data)

        self.assertTrue(a == b)

    def test_neq(self):
        data = 3
        data2 = 5
        a = mynp.array(data)
        b = mynp.array(data2)
        self.assertTrue(a != 5)
        self.assertTrue(a != b)

        data = [1, 2, 3]
        data2 = [4, 5, 6]
        a = mynp.array(data)
        b = mynp.array(data2)

        self.assertTrue(a != b)
        self.assertTrue(a != 0)

        data = [
            [1, 2],
            [3, 4]
        ]
        data2 = [
            [-1, -2],
            [-3, -4]
        ]
        a = mynp.array(data)
        b = mynp.array(data2)

        self.assertTrue(a != b)
        self.assertTrue(a != 0)

        data = [
            [
                [1, -2],
                [-3, 4]
            ],
            [
                [-5, 6],
                [7, -8]
            ]
        ]
        data2 = [
            [
                [-1, 2],
                [3, -4]
            ],
            [
                [5, -6],
                [-7, 8]
            ]
        ]
        a = mynp.array(data)
        b = mynp.array(data2)

        self.assertTrue(a != b)
        self.assertTrue(a != 0)

    def test_ndim(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual(a.ndim, 0)

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual(a.ndim, 1)

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual(a.ndim, 2)

        data = [
            [
                [1, -2],
                [-3, 4]
            ],
            [
                [-5, 6],
                [7, -8]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual(a.ndim, 3)

    def test_shape(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual(a.shape, ())

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual(a.shape, (3,))

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual(a.shape, (2, 2))

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual(a.shape, (4, 2, 3))

    def test_len(self):
        data = 3
        a = mynp.array(data)

        with self.assertRaises(TypeError):
            self.assertEqual(len(a), 0)

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual(len(a), 3)

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual(len(a), 2)

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual(len(a), 4)

    def test_size(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual(a.size, 1)

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual(a.size, 3)

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual(a.size, 4)

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual(a.size, 24)

    def test_flatten(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual(a.flatten().data, [3])

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual(a.flatten().data, [1, 2, 3])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual(a.flatten().data, [1, 2, 3, 4])

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual(a.flatten().data, [1, -2, 3, -4, 5, -6, -7, 8, -9, 10, -11, 12, -1, 2, -3, 4, -5, 6, 7, -8, 9, -10, 11, -12])

    def test_reshape(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual(a.reshape(1).data, [3])
        self.assertEqual(a.reshape(1, 1).data, [[3]])
        self.assertEqual(a.reshape(1, 1, 1).data, [[[3]]])

        data = [3]
        a = mynp.array(data)

        self.assertEqual(a.reshape(1).data, [3])
        self.assertEqual(a.reshape(1, 1).data, [[3]])
        self.assertEqual(a.reshape(1, 1, 1).data, [[[3]]])

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual(a.reshape((3, 1)).data, [[1], [2], [3]])
        self.assertEqual(a.reshape(3, 1).data, [[1], [2], [3]])
        self.assertEqual(a.reshape((-1, 1)).data, [[1], [2], [3]])
        self.assertEqual(a.reshape(-1, 1).data, [[1], [2], [3]])

        with self.assertRaises(ValueError):
            a.reshape((-1, 5))
        with self.assertRaises(ValueError):
            a.reshape((2, 3))

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual(a.reshape((1, 4)).data, [[1, 2, 3, 4]])
        self.assertEqual(a.reshape(1, 4).data, [[1, 2, 3, 4]])
        self.assertEqual(a.reshape((-1, 4)).data, [[1, 2, 3, 4]])
        self.assertEqual(a.reshape(-1, 4).data, [[1, 2, 3, 4]])
        self.assertEqual(a.reshape((4, 1)).data, [[1], [2], [3], [4]])
        self.assertEqual(a.reshape(4, 1).data, [[1], [2], [3], [4]])
        self.assertEqual(a.reshape((-1, 1)).data, [[1], [2], [3], [4]])
        self.assertEqual(a.reshape(-1, 1).data, [[1], [2], [3], [4]])

        with self.assertRaises(ValueError):
            a.reshape((-1, 3))
        with self.assertRaises(ValueError):
            a.reshape((8, 7))

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual(a.reshape((2, 3, 4)).data, [
            [
                [1, -2, 3, -4], [5, -6, -7, 8], [-9, 10, -11, 12]
            ],
            [
                [-1, 2, -3, 4], [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])
        self.assertEqual(a.reshape(2, 3, 4).data, [
            [
                [1, -2, 3, -4], [5, -6, -7, 8], [-9, 10, -11, 12]
            ],
            [
                [-1, 2, -3, 4], [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])
        self.assertEqual(a.reshape((-1, 3, 4)).data, [
            [
                [1, -2, 3, -4], [5, -6, -7, 8], [-9, 10, -11, 12]
            ],
            [
                [-1, 2, -3, 4], [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])
        self.assertEqual(a.reshape(-1, 3, 4).data, [
            [
                [1, -2, 3, -4], [5, -6, -7, 8], [-9, 10, -11, 12]
            ],
            [
                [-1, 2, -3, 4], [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])

        self.assertEqual(a.reshape((4, 3, 2)).data, [
            [
                [1, -2], [3, -4], [5, -6]
            ],
            [
                [-7, 8], [-9, 10], [-11, 12]
            ],
            [
                [-1, 2], [-3, 4], [-5, 6]
            ],
            [
                [7, -8], [9, -10], [11, -12]
            ]
        ])
        self.assertEqual(a.reshape(4, 3, 2).data, [
            [
                [1, -2], [3, -4], [5, -6]
            ],
            [
                [-7, 8], [-9, 10], [-11, 12]
            ],
            [
                [-1, 2], [-3, 4], [-5, 6]
            ],
            [
                [7, -8], [9, -10], [11, -12]
            ]
        ])
        self.assertEqual(a.reshape((-1, 3, 2)).data, [
            [
                [1, -2], [3, -4], [5, -6]
            ],
            [
                [-7, 8], [-9, 10], [-11, 12]
            ],
            [
                [-1, 2], [-3, 4], [-5, 6]
            ],
            [
                [7, -8], [9, -10], [11, -12]
            ]
        ])
        self.assertEqual(a.reshape(-1, 3, 2).data, [
            [
                [1, -2], [3, -4], [5, -6]
            ],
            [
                [-7, 8], [-9, 10], [-11, 12]
            ],
            [
                [-1, 2], [-3, 4], [-5, 6]
            ],
            [
                [7, -8], [9, -10], [11, -12]
            ]
        ])

        self.assertEqual(a.reshape((3, 4, 2)).data, [
            [
                [1, -2], [3, -4], [5, -6], [-7, 8]
            ],
            [
                [-9, 10], [-11, 12], [-1, 2], [-3, 4]
            ],
            [
                [-5, 6], [7, -8], [9, -10], [11, -12]
            ]
        ])
        self.assertEqual(a.reshape(3, 4, 2).data, [
            [
                [1, -2], [3, -4], [5, -6], [-7, 8]
            ],
            [
                [-9, 10], [-11, 12], [-1, 2], [-3, 4]
            ],
            [
                [-5, 6], [7, -8], [9, -10], [11, -12]
            ]
        ])
        self.assertEqual(a.reshape((-1, 4, 2)).data, [
            [
                [1, -2], [3, -4], [5, -6], [-7, 8]
            ],
            [
                [-9, 10], [-11, 12], [-1, 2], [-3, 4]
            ],
            [
                [-5, 6], [7, -8], [9, -10], [11, -12]
            ]
        ])
        self.assertEqual(a.reshape(-1, 4, 2).data, [
            [
                [1, -2], [3, -4], [5, -6], [-7, 8]
            ],
            [
                [-9, 10], [-11, 12], [-1, 2], [-3, 4]
            ],
            [
                [-5, 6], [7, -8], [9, -10], [11, -12]
            ]
        ])

        self.assertEqual(a.reshape((3, 2, 4)).data, [
            [
                [1, -2, 3, -4], [5, -6, -7, 8]
            ],
            [
                [-9, 10, -11, 12], [-1, 2, -3, 4]
            ],
            [
                [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])
        self.assertEqual(a.reshape(3, 2, 4).data, [
            [
                [1, -2, 3, -4], [5, -6, -7, 8]
            ],
            [
                [-9, 10, -11, 12], [-1, 2, -3, 4]
            ],
            [
                [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])
        self.assertEqual(a.reshape((-1, 2, 4)).data, [
            [
                [1, -2, 3, -4], [5, -6, -7, 8]
            ],
            [
                [-9, 10, -11, 12], [-1, 2, -3, 4]
            ],
            [
                [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])
        self.assertEqual(a.reshape(-1, 2, 4).data, [
            [
                [1, -2, 3, -4], [5, -6, -7, 8]
            ],
            [
                [-9, 10, -11, 12], [-1, 2, -3, 4]
            ],
            [
                [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])

        with self.assertRaises(ValueError):
            a.reshape((-1, 3, 3))
        with self.assertRaises(ValueError):
            a.reshape((2, 5, 2))

    def test_T(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual(a.T.data, 3)

        data = [3]
        a = mynp.array(data)

        self.assertEqual(a.T.data, [3])

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual(a.T.data, [1, 2, 3])

        data = [[1], [2], [3]]
        a = mynp.array(data)

        self.assertEqual(a.T.data, [[1, 2, 3]])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual(a.T.data, [
            [1, 3],
            [2, 4]
        ])

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual(a.T.data, [
            [
                [1, -7, -1, 7],
                [-4, 10, 4, -10]
            ],
            [
                [-2, 8, 2, -8],
                [5, -11, -5, 11]
            ],
            [
                [3, -9, -3, 9],
                [-6, 12, 6, -12]
            ]
        ])

    def test_zeros(self):
        a = mynp.zeros(3)

        self.assertEqual(a.data, [0, 0, 0])

        a = mynp.zeros([3])

        self.assertEqual(a.data, [0, 0, 0])

        a = mynp.zeros((3))

        self.assertEqual(a.data, [0, 0, 0])

        a = mynp.zeros((2, 2))

        self.assertEqual(a.data, [
            [0, 0],
            [0, 0]
        ])

        a = mynp.zeros((4, 2, 3))

        self.assertEqual(a.data, [
            [
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0]
            ]
        ])

    def test_item(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual(a.item(), 3)

        data = [3]
        a = mynp.array(data)

        self.assertEqual(a.item(), 3)

        data = [[3]]
        a = mynp.array(data)

        self.assertEqual(a.item(), 3)

        with self.assertRaises(ValueError):
            data = [1, 2]
            a = mynp.array(data)
            a.item()

    def test_zeros_like(self):
        data = 3
        a = mynp.zeros_like(data)

        self.assertEqual(a.data, 0)

        data = [3]
        a = mynp.zeros_like(data)

        self.assertEqual(a.data, [0])

        data = [1, 2, 3]
        a = mynp.zeros_like(data)

        self.assertEqual(a.data, [0, 0, 0])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.zeros_like(data)

        self.assertEqual(a.data, [
            [0, 0],
            [0, 0]
        ])

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.zeros_like(data)

        self.assertEqual(a.data, [
            [
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0]
            ],
            [
                [0, 0, 0],
                [0, 0, 0]
            ]
        ])

    def test_ones(self):
        a = mynp.ones(3)

        self.assertEqual(a.data, [1, 1, 1])

        a = mynp.ones([3])

        self.assertEqual(a.data, [1, 1, 1])

        a = mynp.ones((3))

        self.assertEqual(a.data, [1, 1, 1])

        a = mynp.ones((2, 2))

        self.assertEqual(a.data, [
            [1, 1],
            [1, 1]
        ])

        a = mynp.ones((4, 2, 3))

        self.assertEqual(a.data, [
            [
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [1, 1, 1],
                [1, 1, 1]
            ]
        ])

    def test_ones_like(self):
        data = 3
        a = mynp.ones_like(data)

        self.assertEqual(a.data, 1)

        data = [3]
        a = mynp.ones_like(data)

        self.assertEqual(a.data, [1])

        data = [1, 2, 3]
        a = mynp.ones_like(data)

        self.assertEqual(a.data, [1, 1, 1])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.ones_like(data)

        self.assertEqual(a.data, [
            [1, 1],
            [1, 1]
        ])

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.ones_like(data)

        self.assertEqual(a.data, [
            [
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [1, 1, 1],
                [1, 1, 1]
            ],
            [
                [1, 1, 1],
                [1, 1, 1]
            ]
        ])

    def test_add(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual((a + 5).data, 8)

        data = [3]
        a = mynp.array(data)

        self.assertEqual((a + 5).data, [8])

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual((a + 5).data, [6, 7, 8])

        data2 = [-3, -2, -1]
        b = mynp.array(data2)

        self.assertEqual((a + b).data, [-2, 0, 2])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual((a + 5).data, [
            [6, 7],
            [8, 9]
        ])

        data2 = [
            [1, 0],
            [0, 2]
        ]
        b = mynp.array(data2)

        self.assertEqual((a + b).data, [
            [2, 2],
            [3, 6]
        ])

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual((a + 5).data, [
            [
                [6, 3, 8],
                [1, 10, -1]
            ],
            [
                [-2, 13, -4],
                [15, -6, 17]
            ],
            [
                [4, 7, 2],
                [9, 0, 11]
            ],
            [
                [12, -3, 14],
                [-5, 16, -7]
            ]
        ])

        data2 = [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ],
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ]
        b = mynp.array(data2)

        self.assertEqual((a + b).data, [
            [
                [2, 0, 6],
                [0, 10, 0]
            ],
            [
                [0, 16, 0],
                [20, 0, 24]
            ],
            [
                [0, 4, 0],
                [8, 0, 12]
            ],
            [
                [14, 0, 18],
                [0, 22, 0]
            ]
        ])

        a = mynp.array([
            [1],
            [2]
        ])

        b = mynp.array([
            [
                [1, 2, 3]
            ],
            [
                [4, 5, 6]
            ],
            [
                [7, 8, 9]
            ],
            [
                [10, 11, 12]
            ]
        ])

        self.assertEqual((a + b).data, [
            [
                [2, 3, 4],
                [3, 4, 5]
            ],
            [
                [5, 6, 7],
                [6, 7, 8]
            ],
            [
                [8, 9, 10],
                [9, 10, 11]
            ],
            [
                [11, 12, 13],
                [12, 13, 14]
            ]
        ])

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = mynp.array(data)
            data2 = [1, 2]
            b = mynp.array(data2)
            a + b

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = mynp.array(data)
            data2 = [1, 2, 3, 4, 5]
            b = mynp.array(data2)
            a + b

        with self.assertRaises(ValueError):
            a = mynp.array([
                [1, 2],
                [3, 4]
            ])
            b = mynp.array([1, 2, 3])
            a + b

    def test_radd(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual((5 + a).data, 8)

        data = [3]
        a = mynp.array(data)

        self.assertEqual((5 + a).data, [8])

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual((5 + a).data, [6, 7, 8])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual((5 + a).data, [
            [6, 7],
            [8, 9]
        ])

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual((5 + a).data, [
            [
                [6, 3, 8],
                [1, 10, -1]
            ],
            [
                [-2, 13, -4],
                [15, -6, 17]
            ],
            [
                [4, 7, 2],
                [9, 0, 11]
            ],
            [
                [12, -3, 14],
                [-5, 16, -7]
            ]
        ])

    def test_sub(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual((a - 5).data, -2)

        data = [3]
        a = mynp.array(data)

        self.assertEqual((a - 5).data, [-2])

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual((a - 5).data, [-4, -3, -2])

        data2 = [-3, -2, -1]
        b = mynp.array(data2)

        self.assertEqual((a - b).data, [4, 4, 4])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual((a - 5).data, [
            [-4, -3],
            [-2, -1]
        ])

        data2 = [
            [1, 0],
            [0, 2]
        ]
        b = mynp.array(data2)

        self.assertEqual((a - b).data, [
            [0, 2],
            [3, 2]
        ])

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual((a - 5).data, [
            [
                [-4, -7, -2],
                [-9, 0, -11]
            ],
            [
                [-12, 3, -14],
                [5, -16, 7]
            ],
            [
                [-6, -3, -8],
                [-1, -10, 1]
            ],
            [
                [2, -13, 4],
                [-15, 6, -17]
            ]
        ])

        data2 = [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ],
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ]
        b = mynp.array(data2)

        self.assertEqual((a - b).data, [
            [
                [0, -4, 0],
                [-8, 0, -12]
            ],
            [
                [-14, 0, -18],
                [0, -22, 0]
            ],
            [
                [-2, 0, -6],
                [0, -10, 0]
            ],
            [
                [0, -16, 0],
                [-20, 0, -24]
            ]
        ])

        a = mynp.array([
            [1],
            [2]
        ])

        b = mynp.array([
            [
                [1, 2, 3]
            ],
            [
                [4, 5, 6]
            ],
            [
                [7, 8, 9]
            ],
            [
                [10, 11, 12]
            ]
        ])

        self.assertEqual((a - b).data, [
            [
                [0, -1, -2],
                [1, 0, -1]
            ],
            [
                [-3, -4, -5],
                [-2, -3, -4]
            ],
            [
                [-6, -7, -8],
                [-5, -6, -7]
            ],
            [
                [-9, -10, -11],
                [-8, -9, -10]
            ]
        ])

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = mynp.array(data)
            data2 = [1, 2]
            b = mynp.array(data2)
            a - b

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = mynp.array(data)
            data2 = [1, 2, 3, 4, 5]
            b = mynp.array(data2)
            a - b

        with self.assertRaises(ValueError):
            a = mynp.array([
                [1, 2],
                [3, 4]
            ])
            b = mynp.array([1, 2, 3])
            a + b

    def test_mul(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual((a * 5).data, 15)

        data = [3]
        a = mynp.array(data)

        self.assertEqual((a * 5).data, [15])

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual((a * 5).data, [5, 10, 15])

        data2 = [-3, -2, -1]
        b = mynp.array(data2)

        self.assertEqual((a * b).data, [-3, -4, -3])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual((a * 5).data, [
            [5, 10],
            [15, 20]
        ])

        data2 = [
            [1, 0],
            [0, 2]
        ]
        b = mynp.array(data2)

        self.assertEqual((a * b).data, [
            [1, 0],
            [0, 8]
        ])

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual((a * 5).data, [
            [
                [5, -10, 15],
                [-20, 25, -30]
            ],
            [
                [-35, 40, -45],
                [50, -55, 60]
            ],
            [
                [-5, 10, -15],
                [20, -25, 30]
            ],
            [
                [35, -40, 45],
                [-50, 55, -60]
            ]
        ])

        data2 = [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ],
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ]
        b = mynp.array(data2)

        self.assertEqual((a * b).data, [
            [
                [1, -4, 9],
                [-16, 25, -36]
            ],
            [
                [-49, 64, -81],
                [100, -121, 144]
            ],
            [
                [-1, 4, -9],
                [16, -25, 36]
            ],
            [
                [49, -64, 81],
                [-100, 121, -144]
            ]
        ])

        a = mynp.array([
            [1],
            [2]
        ])

        b = mynp.array([
            [
                [1, 2, 3]
            ],
            [
                [4, 5, 6]
            ],
            [
                [7, 8, 9]
            ],
            [
                [10, 11, 12]
            ]
        ])

        self.assertEqual((a * b).data, [
            [
                [1, 2, 3],
                [2, 4, 6]
            ],
            [
                [4, 5, 6],
                [8, 10, 12]
            ],
            [
                [7, 8, 9],
                [14, 16, 18]
            ],
            [
                [10, 11, 12],
                [20, 22, 24]
            ]
        ])

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = mynp.array(data)
            data2 = [1, 2]
            b = mynp.array(data2)
            a * b

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = mynp.array(data)
            data2 = [1, 2, 3, 4, 5]
            b = mynp.array(data2)
            a * b

        with self.assertRaises(ValueError):
            a = mynp.array([
                [1, 2],
                [3, 4]
            ])
            b = mynp.array([1, 2, 3])
            a * b

    def test_truediv(self):
        data = 3
        a = mynp.array(data)

        self.assertEqual((a / 5).data, 0.6)

        data = [3]
        a = mynp.array(data)

        self.assertEqual((a / 5).data, [0.6])

        data = [1, 2, 3]
        a = mynp.array(data)

        self.assertEqual((a / 5).data, [0.2, 0.4, 0.6])

        data2 = [-3, -2, -1]
        b = mynp.array(data2)

        self.assertEqual((a / b).data, [-0.3333333333333333, -1.0, -3.0])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        self.assertEqual((a / 5).data, [
            [0.2, 0.4],
            [0.6, 0.8]
        ])

        data2 = [
            [1, -1],
            [-1, 2]
        ]
        b = mynp.array(data2)

        self.assertEqual((a / b).data, [
            [1.0, -2.0],
            [-3.0, 2.0]
        ])

        data = [
            [
                [1, -2, 3],
                [-4, 5, -6]
            ],
            [
                [-7, 8, -9],
                [10, -11, 12]
            ],
            [
                [-1, 2, -3],
                [4, -5, 6]
            ],
            [
                [7, -8, 9],
                [-10, 11, -12]
            ]
        ]
        a = mynp.array(data)

        self.assertEqual((a / 5).data, [
            [
                [0.2, -0.4, 0.6],
                [-0.8, 1.0, -1.2]
            ],
            [
                [-1.4, 1.6, -1.8],
                [2.0, -2.2, 2.4]
            ],
            [
                [-0.2, 0.4, -0.6],
                [0.8, -1.0, 1.2]
            ],
            [
                [1.4, -1.6, 1.8],
                [-2.0, 2.2, -2.4]
            ]
        ])

        data2 = [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ],
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ]
        b = mynp.array(data2)

        self.assertEqual((a / b).data, [
            [
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0]
            ],
            [
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0]
            ],
            [
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0]
            ],
            [
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0]
            ]
        ])

        a = mynp.array([
            [1],
            [2]
        ])

        b = mynp.array([
            [
                [1, 2, 3]
            ],
            [
                [4, 5, 6]
            ],
            [
                [7, 8, 9]
            ],
            [
                [10, 11, 12]
            ]
        ])

        self.assertEqual((a / b).data, [
            [
                [1.0, 0.5, 0.3333333333333333],
                [2.0, 1.0, 0.6666666666666666]
            ],
            [
                [0.25, 0.2, 0.16666666666666666],
                [0.5, 0.4, 0.3333333333333333]
            ],
            [
                [0.14285714285714285, 0.125, 0.1111111111111111],
                [0.2857142857142857, 0.25, 0.2222222222222222]
            ],
            [
                [0.1, 0.09090909090909091, 0.08333333333333333],
                [0.2, 0.18181818181818182, 0.16666666666666666]
            ]
        ])

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = mynp.array(data)
            data2 = [1, 2]
            b = mynp.array(data2)
            a / b

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = mynp.array(data)
            data2 = [1, 2, 3, 4, 5]
            b = mynp.array(data2)
            a / b

        with self.assertRaises(ValueError):
            a = mynp.array([
                [1, 2],
                [3, 4]
            ])
            b = mynp.array([1, 2, 3])
            a / b

    def test_matmul(self):
        data = [1, 2]
        a = mynp.array(data)

        data2 = [
            [1, 2],
            [3, 4]
        ]
        b = mynp.array(data2)

        self.assertEqual((a @ b).data, [7, 10])
        self.assertEqual((b @ a).data, [5, 11])

        data3 = [3, 4]
        c = mynp.array(data3)

        self.assertEqual((a @ c).data, 11)
        self.assertEqual((c @ a).data, 11)

        data4 = [[1], [2]]
        d = mynp.array(data4)

        with self.assertRaises(ValueError):
            d @ b
        self.assertEqual((b @ d).data, [[5], [11]])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = mynp.array(data)

        data2 = [
            [-1, 0],
            [4, -5]
        ]
        b = mynp.array(data2)

        self.assertEqual((a @ b).data, [
            [7, -10],
            [13, -20]
        ])

        data = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
        a = mynp.array(data)

        data2 = [
            [-1, 0, 3, 1],
            [4, -5, -2, 3]
        ]
        b = mynp.array(data2)

        self.assertEqual((a @ b).data, [
            [7, -10, -1, 7],
            [13, -20, 1, 15],
            [19, -30, 3, 23]
        ])

        with self.assertRaises(ValueError):
            b @ a

    def test_einsum(self):
        a = mynp.array([
            [1, 2],
            [3, 4]
        ])

        self.assertEqual(mynp.einsum('ii->', a).data, 5)

        a = mynp.array([
            [1, 2],
            [3, 4]
        ])

        self.assertEqual(mynp.einsum('ij->ij', a).data, [
            [1, 2],
            [3, 4]
        ])
        self.assertEqual(mynp.einsum('ij->ji', a).data, [
            [1, 3],
            [2, 4]
        ])

        a = mynp.array([
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ],
        ])

        self.assertEqual(mynp.einsum('ijk->jki', a).data, [
            [
                [1, 5],
                [2, 6]
            ],
            [
                [3, 7],
                [4, 8]
            ]
        ])


        a = mynp.array([1, 2, 3])
        b = mynp.array([-2, 1, 4])

        self.assertEqual(mynp.einsum('i,i->', a, b).data, 12)

        a = mynp.array([
            [1, 2],
            [3, 4]
        ])
        b = mynp.array([
            [-5, 6],
            [7, -8]
        ])

        self.assertEqual(mynp.einsum('ij,ij->', a, b).data, -4)
        self.assertEqual(mynp.einsum('ij,ji->', a, b).data, -5)

        a = mynp.array([1, 2])

        b = mynp.array([
            [1, 2],
            [3, 4]
        ])

        self.assertEqual(mynp.einsum('i,ij->j', a, b).data, [7, 10])
        self.assertEqual(mynp.einsum('i,ji->j', a, b).data, [5, 11])

        self.assertEqual(mynp.einsum('ij,i->j', b, a).data, [7, 10])
        self.assertEqual(mynp.einsum('ij,j->i', b, a).data, [5, 11])

        a = mynp.array([
            [1, 2],
            [3, 4]
        ])

        b = mynp.array([
            [-2, 1],
            [-5, 3]
        ])

        self.assertEqual(mynp.einsum('ij,jk->ik', a, b).data, [
            [-12, 7],
            [-26, 15]
        ])

        self.assertEqual(mynp.einsum('jk,ki->ji', a, b).data, [
            [-12, 7],
            [-26, 15]
        ])

        a = mynp.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])

        b = mynp.array([
            [7, 8, 9, 10],
            [11, 12, 13, 14]
        ])

        self.assertEqual(mynp.einsum('ij,jk->ik', a, b).data, [
            [ 29, 32, 35, 38],
            [ 65, 72, 79, 86],
            [101, 112, 123, 134]
        ])

        a = mynp.array([
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ],
        ])

        b = mynp.array([
            [
                [-1, -5],
                [-3, 2],
                [1, 4],
            ],
            [
                [3, 6],
                [-3, 2],
                [-4, 1],
            ]
        ])

        self.assertEqual(mynp.einsum('ijk,ilj->jl', a, b).data, [
            [30, -42, -41],
            [55, 44, 43]
        ])

        with self.assertRaises(ValueError):
            a = mynp.array([
                [1, 2],
                [3, 4]
            ])
            b = mynp.array([
                [-2, 1],
                [-5, 3]
            ])
            mynp.einsum('ijl,jk->ik', a, b)

        with self.assertRaises(ValueError):
            a = mynp.array([
                [1, 2],
                [3, 4]
            ])
            b = mynp.array([
                [-2, 1],
                [-5, 3]
            ])
            mynp.einsum('i,jk->ik', a, b)
