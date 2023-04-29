import unittest
import mynumpy as mynp

class TestNdArray(unittest.TestCase):
    def setUp(self):
        ...

    def tearDown(self):
        ...

    def test_create(self):
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

    def test_zeros_like(self):
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

    def test_add(self):
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

    def test_sub(self):
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

    def test_mul(self):
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

    def test_truediv(self):
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
