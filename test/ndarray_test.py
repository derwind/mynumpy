import unittest

import mynumpy as xp


class TestNdArray(unittest.TestCase):
    def setUp(self):
        ...

    def tearDown(self):
        ...

    def test_create_ndarray(self):
        a = xp.ndarray(3)
        self.assertEqual(len(a.tolist()), 3)

        a = xp.ndarray((3,))
        self.assertEqual(len(a.tolist()), 3)

    def test_create(self):
        data = 3
        a = xp.array(data)
        self.assertEqual(data, a.tolist())

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual(data, a.tolist())

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        self.assertEqual(data, a.tolist())

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
        a = xp.array(data)

        self.assertEqual(data, a.tolist())

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
        a = xp.array(data)

        self.assertEqual(data, a.tolist())

        a = xp.array(data, dtype=float)

        answer = [
            [
                [1., -2.],
                [-3., 4.]
            ],
            [
                [-5., 6.],
                [7., -8.]
            ]
        ]

        self.assertEqual(answer, a.tolist())

        a = xp.array(data, dtype=complex)

        answer = [
            [
                [1. + 0j, -2. + 0j],
                [-3. + 0j, 4. + 0j]
            ],
            [
                [-5. + 0j, 6. + 0j],
                [7. + 0j, -8. + 0j]
            ]
        ]

        self.assertEqual(answer, a.tolist())

    def test_eq(self):
        data = 3
        a = xp.array(data)
        b = xp.array(data)
        self.assertTrue(a == 3)
        self.assertTrue(a == b)

        data = [1, 2, 3]
        a = xp.array(data)
        b = xp.array(data)

        self.assertTrue(a.tolist() == b.tolist())

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)
        b = xp.array(data)

        self.assertTrue(a.tolist() == b.tolist())

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
        a = xp.array(data)
        b = xp.array(data)

        self.assertTrue(a.tolist() == b.tolist())

    def test_neq(self):
        data = 3
        data2 = 5
        a = xp.array(data)
        b = xp.array(data2)
        self.assertTrue(a.tolist() != 5)
        self.assertTrue(a.tolist() != b.tolist())

        data = [1, 2, 3]
        data2 = [4, 5, 6]
        a = xp.array(data)
        b = xp.array(data2)

        self.assertTrue(a.tolist() != b.tolist())
        self.assertTrue(a.tolist() != 0)

        data = [
            [1, 2],
            [3, 4]
        ]
        data2 = [
            [-1, -2],
            [-3, -4]
        ]
        a = xp.array(data)
        b = xp.array(data2)

        self.assertTrue(a.tolist() != b.tolist())
        self.assertTrue(a.tolist() != 0)

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
        a = xp.array(data)
        b = xp.array(data2)

        self.assertTrue(a.tolist() != b.tolist())
        self.assertTrue(a.tolist() != 0)

    def test_ndim(self):
        data = 3
        a = xp.array(data)

        self.assertEqual(a.ndim, 0)

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual(a.ndim, 1)

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

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
        a = xp.array(data)

        self.assertEqual(a.ndim, 3)

    def test_shape(self):
        data = 3
        a = xp.array(data)

        self.assertEqual(a.shape, ())

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual(a.shape, (3,))

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

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
        a = xp.array(data)

        self.assertEqual(a.shape, (4, 2, 3))

    def test_len(self):
        data = 3
        a = xp.array(data)

        with self.assertRaises(TypeError):
            self.assertEqual(len(a), 0)

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual(len(a), 3)

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

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
        a = xp.array(data)

        self.assertEqual(len(a), 4)

    def test_size(self):
        data = 3
        a = xp.array(data)

        self.assertEqual(a.size, 1)

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual(a.size, 3)

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

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
        a = xp.array(data)

        self.assertEqual(a.size, 24)

    def test_flatten(self):
        data = 3
        a = xp.array(data)

        self.assertEqual(a.flatten().tolist(), [3])

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual(a.flatten().tolist(), [1, 2, 3])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        self.assertEqual(a.flatten().tolist(), [1, 2, 3, 4])

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
        a = xp.array(data)

        self.assertEqual(a.flatten().tolist(), [1, -2, 3, -4, 5, -6, -7, 8, -9, 10, -11, 12, -1, 2, -3, 4, -5, 6, 7, -8, 9, -10, 11, -12])

    def test_reshape(self):
        data = 3
        a = xp.array(data)

        self.assertEqual(a.reshape(1).tolist(), [3])
        self.assertEqual(a.reshape(1, 1).tolist(), [[3]])
        self.assertEqual(a.reshape(1, 1, 1).tolist(), [[[3]]])

        data = [3]
        a = xp.array(data)

        self.assertEqual(a.reshape(1).tolist(), [3])
        self.assertEqual(a.reshape(1, 1).tolist(), [[3]])
        self.assertEqual(a.reshape(1, 1, 1).tolist(), [[[3]]])

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual(a.reshape((3, 1)).tolist(), [[1], [2], [3]])
        self.assertEqual(a.reshape(3, 1).tolist(), [[1], [2], [3]])
        self.assertEqual(a.reshape((-1, 1)).tolist(), [[1], [2], [3]])
        self.assertEqual(a.reshape(-1, 1).tolist(), [[1], [2], [3]])

        with self.assertRaises(ValueError):
            a.reshape((-1, 5))
        with self.assertRaises(ValueError):
            a.reshape((2, 3))

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        self.assertEqual(a.reshape((1, 4)).tolist(), [[1, 2, 3, 4]])
        self.assertEqual(a.reshape(1, 4).tolist(), [[1, 2, 3, 4]])
        self.assertEqual(a.reshape((-1, 4)).tolist(), [[1, 2, 3, 4]])
        self.assertEqual(a.reshape(-1, 4).tolist(), [[1, 2, 3, 4]])
        self.assertEqual(a.reshape((4, 1)).tolist(), [[1], [2], [3], [4]])
        self.assertEqual(a.reshape(4, 1).tolist(), [[1], [2], [3], [4]])
        self.assertEqual(a.reshape((-1, 1)).tolist(), [[1], [2], [3], [4]])
        self.assertEqual(a.reshape(-1, 1).tolist(), [[1], [2], [3], [4]])

        with self.assertRaises(ValueError):
            a.reshape((-1, 3))
        with self.assertRaises(ValueError):
            a.reshape((8, 7))

        data = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        a = xp.array(data)

        self.assertEqual(a.reshape((2, 1, 3)).tolist(), [
            [
                [1, 2, 3]
            ],
            [
                [4, 5, 6]
            ]
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
        a = xp.array(data)

        self.assertEqual(a.reshape((2, 3, 4)).tolist(), [
            [
                [1, -2, 3, -4], [5, -6, -7, 8], [-9, 10, -11, 12]
            ],
            [
                [-1, 2, -3, 4], [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])
        self.assertEqual(a.reshape(2, 3, 4).tolist(), [
            [
                [1, -2, 3, -4], [5, -6, -7, 8], [-9, 10, -11, 12]
            ],
            [
                [-1, 2, -3, 4], [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])
        self.assertEqual(a.reshape((-1, 3, 4)).tolist(), [
            [
                [1, -2, 3, -4], [5, -6, -7, 8], [-9, 10, -11, 12]
            ],
            [
                [-1, 2, -3, 4], [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])
        self.assertEqual(a.reshape(-1, 3, 4).tolist(), [
            [
                [1, -2, 3, -4], [5, -6, -7, 8], [-9, 10, -11, 12]
            ],
            [
                [-1, 2, -3, 4], [-5, 6, 7, -8], [9, -10, 11, -12]
            ]
        ])

        self.assertEqual(a.reshape((4, 3, 2)).tolist(), [
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
        self.assertEqual(a.reshape(4, 3, 2).tolist(), [
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
        self.assertEqual(a.reshape((-1, 3, 2)).tolist(), [
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
        self.assertEqual(a.reshape(-1, 3, 2).tolist(), [
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

        self.assertEqual(a.reshape((3, 4, 2)).tolist(), [
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
        self.assertEqual(a.reshape(3, 4, 2).tolist(), [
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
        self.assertEqual(a.reshape((-1, 4, 2)).tolist(), [
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
        self.assertEqual(a.reshape(-1, 4, 2).tolist(), [
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

        self.assertEqual(a.reshape((3, 2, 4)).tolist(), [
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
        self.assertEqual(a.reshape(3, 2, 4).tolist(), [
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
        self.assertEqual(a.reshape((-1, 2, 4)).tolist(), [
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
        self.assertEqual(a.reshape(-1, 2, 4).tolist(), [
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
        a = xp.array(data)

        self.assertEqual(a.T.tolist(), 3)

        data = [3]
        a = xp.array(data)

        self.assertEqual(a.T.tolist(), [3])

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual(a.T.tolist(), [1, 2, 3])

        data = [[1], [2], [3]]
        a = xp.array(data)

        self.assertEqual(a.T.tolist(), [[1, 2, 3]])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        self.assertEqual(a.T.tolist(), [
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
        a = xp.array(data)

        self.assertEqual(a.T.tolist(), [
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
        a = xp.zeros(3)

        self.assertEqual(a.tolist(), [0, 0, 0])

        a = xp.zeros([3])

        self.assertEqual(a.tolist(), [0, 0, 0])

        a = xp.zeros((3))

        self.assertEqual(a.tolist(), [0, 0, 0])

        a = xp.zeros((2, 2))

        self.assertEqual(a.tolist(), [
            [0, 0],
            [0, 0]
        ])

        a = xp.zeros((4, 2, 3))

        self.assertEqual(a.tolist(), [
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
        a = xp.array(data)

        self.assertEqual(a.item(), 3)

        data = [3]
        a = xp.array(data)

        self.assertEqual(a.item(), 3)

        data = [[3]]
        a = xp.array(data)

        self.assertEqual(a.item(), 3)

        with self.assertRaises(ValueError):
            data = [1, 2]
            a = xp.array(data)
            a.item()

    def test_zeros_like(self):
        data = 3
        a = xp.zeros_like(data)

        self.assertEqual(a.tolist(), 0)

        data = [3]
        a = xp.zeros_like(data)

        self.assertEqual(a.tolist(), [0])

        data = [1, 2, 3]
        a = xp.zeros_like(data)

        self.assertEqual(a.tolist(), [0, 0, 0])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.zeros_like(data)

        self.assertEqual(a.tolist(), [
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
        a = xp.zeros_like(data)

        self.assertEqual(a.tolist(), [
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
        a = xp.ones(3)

        self.assertEqual(a.tolist(), [1, 1, 1])

        a = xp.ones([3])

        self.assertEqual(a.tolist(), [1, 1, 1])

        a = xp.ones((3))

        self.assertEqual(a.tolist(), [1, 1, 1])

        a = xp.ones((2, 2))

        self.assertEqual(a.tolist(), [
            [1, 1],
            [1, 1]
        ])

        a = xp.ones((4, 2, 3))

        self.assertEqual(a.tolist(), [
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
        a = xp.ones_like(data)

        self.assertEqual(a.tolist(), 1)

        data = [3]
        a = xp.ones_like(data)

        self.assertEqual(a.tolist(), [1])

        data = [1, 2, 3]
        a = xp.ones_like(data)

        self.assertEqual(a.tolist(), [1, 1, 1])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.ones_like(data)

        self.assertEqual(a.tolist(), [
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
        a = xp.ones_like(data)

        self.assertEqual(a.tolist(), [
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
        a = xp.array(data)

        self.assertEqual((a + 5).tolist(), 8)

        data = [3]
        a = xp.array(data)

        self.assertEqual((a + 5).tolist(), [8])

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual((a + 5).tolist(), [6, 7, 8])

        data2 = [-3, -2, -1]
        b = xp.array(data2)

        self.assertEqual((a + b).tolist(), [-2, 0, 2])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        self.assertEqual((a + 5).tolist(), [
            [6, 7],
            [8, 9]
        ])

        data2 = [
            [1, 0],
            [0, 2]
        ]
        b = xp.array(data2)

        self.assertEqual((a + b).tolist(), [
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
        a = xp.array(data)

        self.assertEqual((a + 5).tolist(), [
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
        b = xp.array(data2)

        self.assertEqual((a + b).tolist(), [
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

        a = xp.array([
            [1],
            [2]
        ])

        b = xp.array([
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

        self.assertEqual((a + b).tolist(), [
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
            a = xp.array(data)
            data2 = [1, 2]
            b = xp.array(data2)
            a + b

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = xp.array(data)
            data2 = [1, 2, 3, 4, 5]
            b = xp.array(data2)
            a + b

        with self.assertRaises(ValueError):
            a = xp.array([
                [1, 2],
                [3, 4]
            ])
            b = xp.array([1, 2, 3])
            a + b

    def test_radd(self):
        data = 3
        a = xp.array(data)

        self.assertEqual((5 + a).tolist(), 8)

        data = [3]
        a = xp.array(data)

        self.assertEqual((5 + a).tolist(), [8])

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual((5 + a).tolist(), [6, 7, 8])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        self.assertEqual((5 + a).tolist(), [
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
        a = xp.array(data)

        self.assertEqual((5 + a).tolist(), [
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
        a = xp.array(data)

        self.assertEqual((a - 5).tolist(), -2)

        data = [3]
        a = xp.array(data)

        self.assertEqual((a - 5).tolist(), [-2])

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual((a - 5).tolist(), [-4, -3, -2])

        data2 = [-3, -2, -1]
        b = xp.array(data2)

        self.assertEqual((a - b).tolist(), [4, 4, 4])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        self.assertEqual((a - 5).tolist(), [
            [-4, -3],
            [-2, -1]
        ])

        data2 = [
            [1, 0],
            [0, 2]
        ]
        b = xp.array(data2)

        self.assertEqual((a - b).tolist(), [
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
        a = xp.array(data)

        self.assertEqual((a - 5).tolist(), [
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
        b = xp.array(data2)

        self.assertEqual((a - b).tolist(), [
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

        a = xp.array([
            [1],
            [2]
        ])

        b = xp.array([
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

        self.assertEqual((a - b).tolist(), [
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
            a = xp.array(data)
            data2 = [1, 2]
            b = xp.array(data2)
            a - b

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = xp.array(data)
            data2 = [1, 2, 3, 4, 5]
            b = xp.array(data2)
            a - b

        with self.assertRaises(ValueError):
            a = xp.array([
                [1, 2],
                [3, 4]
            ])
            b = xp.array([1, 2, 3])
            a + b

    def test_mul(self):
        data = 3
        a = xp.array(data)

        self.assertEqual((a * 5).tolist(), 15)

        data = [3]
        a = xp.array(data)

        self.assertEqual((a * 5).tolist(), [15])

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual((a * 5).tolist(), [5, 10, 15])

        data2 = [-3, -2, -1]
        b = xp.array(data2)

        self.assertEqual((a * b).tolist(), [-3, -4, -3])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        self.assertEqual((a * 5).tolist(), [
            [5, 10],
            [15, 20]
        ])

        data2 = [
            [1, 0],
            [0, 2]
        ]
        b = xp.array(data2)

        self.assertEqual((a * b).tolist(), [
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
        a = xp.array(data)

        self.assertEqual((a * 5).tolist(), [
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
        b = xp.array(data2)

        self.assertEqual((a * b).tolist(), [
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

        a = xp.array([
            [1],
            [2]
        ])

        b = xp.array([
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

        self.assertEqual((a * b).tolist(), [
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
            a = xp.array(data)
            data2 = [1, 2]
            b = xp.array(data2)
            a * b

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = xp.array(data)
            data2 = [1, 2, 3, 4, 5]
            b = xp.array(data2)
            a * b

        with self.assertRaises(ValueError):
            a = xp.array([
                [1, 2],
                [3, 4]
            ])
            b = xp.array([1, 2, 3])
            a * b

    def test_truediv(self):
        data = 3
        a = xp.array(data)

        self.assertEqual((a / 5).tolist(), 0.6)

        data = [3]
        a = xp.array(data)

        self.assertEqual((a / 5).tolist(), [0.6])

        data = [1, 2, 3]
        a = xp.array(data)

        self.assertEqual((a / 5).tolist(), [0.2, 0.4, 0.6])

        data2 = [-3, -2, -1]
        b = xp.array(data2)

        self.assertEqual((a / b).tolist(), [-0.3333333333333333, -1.0, -3.0])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        self.assertEqual((a / 5).tolist(), [
            [0.2, 0.4],
            [0.6, 0.8]
        ])

        data2 = [
            [1, -1],
            [-1, 2]
        ]
        b = xp.array(data2)

        self.assertEqual((a / b).tolist(), [
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
        a = xp.array(data)

        self.assertEqual((a / 5).tolist(), [
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
        b = xp.array(data2)

        self.assertEqual((a / b).tolist(), [
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

        a = xp.array([
            [1],
            [2]
        ])

        b = xp.array([
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

        self.assertEqual((a / b).tolist(), [
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
            a = xp.array(data)
            data2 = [1, 2]
            b = xp.array(data2)
            a / b

        with self.assertRaises(ValueError):
            data = [1, 2, 3]
            a = xp.array(data)
            data2 = [1, 2, 3, 4, 5]
            b = xp.array(data2)
            a / b

        with self.assertRaises(ValueError):
            a = xp.array([
                [1, 2],
                [3, 4]
            ])
            b = xp.array([1, 2, 3])
            a / b

    def test_matmul(self):
        data = [1, 2]
        a = xp.array(data)

        data2 = [
            [1, 2],
            [3, 4]
        ]
        b = xp.array(data2)

        self.assertEqual((a @ b).tolist(), [7, 10])
        self.assertEqual((b @ a).tolist(), [5, 11])

        data3 = [3, 4]
        c = xp.array(data3)

        self.assertEqual((a @ c).tolist(), 11)
        self.assertEqual((c @ a).tolist(), 11)

        data4 = [[1], [2]]
        d = xp.array(data4)

        with self.assertRaises(ValueError):
            d @ b
        self.assertEqual((b @ d).tolist(), [[5], [11]])

        data = [
            [1, 2],
            [3, 4]
        ]
        a = xp.array(data)

        data2 = [
            [-1, 0],
            [4, -5]
        ]
        b = xp.array(data2)

        self.assertEqual((a @ b).tolist(), [
            [7, -10],
            [13, -20]
        ])

        data = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
        a = xp.array(data)

        data2 = [
            [-1, 0, 3, 1],
            [4, -5, -2, 3]
        ]
        b = xp.array(data2)

        self.assertEqual((a @ b).tolist(), [
            [7, -10, -1, 7],
            [13, -20, 1, 15],
            [19, -30, 3, 23]
        ])

        with self.assertRaises(ValueError):
            b @ a

    def test_einsum(self):
        a = xp.array([
            [1, 2],
            [3, 4]
        ])

        self.assertEqual(xp.einsum('ii->', a).tolist(), 5)

        a = xp.array([
            [1, 2],
            [3, 4]
        ])

        self.assertEqual(xp.einsum('ij->ij', a).tolist(), [
            [1, 2],
            [3, 4]
        ])
        self.assertEqual(xp.einsum('ij->ji', a).tolist(), [
            [1, 3],
            [2, 4]
        ])

        a = xp.array([
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ],
        ])

        self.assertEqual(xp.einsum('ijk->jki', a).tolist(), [
            [
                [1, 5],
                [2, 6]
            ],
            [
                [3, 7],
                [4, 8]
            ]
        ])


        a = xp.array([1, 2, 3])
        b = xp.array([-2, 1, 4])

        self.assertEqual(xp.einsum('i,i->', a, b).tolist(), 12)

        a = xp.array([
            [1, 2],
            [3, 4]
        ])
        b = xp.array([
            [-5, 6],
            [7, -8]
        ])

        self.assertEqual(xp.einsum('ij,ij->', a, b).tolist(), -4)
        self.assertEqual(xp.einsum('ij,ji->', a, b).tolist(), -5)
        self.assertEqual(xp.einsum('ij,ij->i', a, b).tolist(), [7, -11])
        self.assertEqual(xp.einsum('ij,ij->j', a, b).tolist(), [16, -20])

        a = xp.array([1, 2])

        b = xp.array([
            [1, 2],
            [3, 4]
        ])

        self.assertEqual(xp.einsum('i,ij->j', a, b).tolist(), [7, 10])
        self.assertEqual(xp.einsum('i,ji->j', a, b).tolist(), [5, 11])

        self.assertEqual(xp.einsum('ij,i->j', b, a).tolist(), [7, 10])
        self.assertEqual(xp.einsum('ij,j->i', b, a).tolist(), [5, 11])

        a = xp.array([
            [1, 2],
            [3, 4]
        ])

        b = xp.array([
            [-2, 1],
            [-5, 3]
        ])

        self.assertEqual(xp.einsum('ij,jk->ik', a, b).tolist(), [
            [-12, 7],
            [-26, 15]
        ])

        self.assertEqual(xp.einsum('jk,ki->ji', a, b).tolist(), [
            [-12, 7],
            [-26, 15]
        ])

        a = xp.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])

        b = xp.array([
            [7, 8, 9, 10],
            [11, 12, 13, 14]
        ])

        self.assertEqual(xp.einsum('ij,jk->ik', a, b).tolist(), [
            [ 29, 32, 35, 38],
            [ 65, 72, 79, 86],
            [101, 112, 123, 134]
        ])

        a = xp.array([
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ],
        ])

        b = xp.array([
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

        self.assertEqual(xp.einsum('ijk,ilj->jl', a, b).tolist(), [
            [30, -42, -41],
            [55, 44, 43]
        ])

        with self.assertRaises((IndexError, ValueError)):
            a = xp.array([
                [1, 2],
                [3, 4],
                [5, 6]
            ])
            xp.einsum('ii->', a)

        with self.assertRaises(ValueError):
            a = xp.array([
                [1, 2],
                [3, 4]
            ])
            b = xp.array([
                [-2, 1],
                [-5, 3]
            ])
            xp.einsum('ijl,jk->ik', a, b)

        with self.assertRaises(ValueError):
            a = xp.array([
                [1, 2],
                [3, 4]
            ])
            b = xp.array([
                [-2, 1],
                [-5, 3]
            ])
            xp.einsum('i,jk->ik', a, b)

    def test_einsum_more_than_2_operands(self):
        a = xp.array([
            [1, 2],
            [3, 4],
            [5, 6],
        ])
        b = xp.array([
            [-2, 1, -1, 3],
            [-5, 3, 0, 2],
        ])
        c = xp.array([
            [0, 2, -3, 0],
            [1, -1, 1, -1],
            [4, -5, 6, -2],
        ])

        self.assertEqual(xp.einsum('ij,jk,lk->il', a, b, c).tolist(), [
            [17, -27, -103],
            [39, -61, -231],
            [61, -95, -359],
        ])

        a = xp.array([
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ],
            [
                [3, -4],
                [-5, 6],
                [1, -2],
            ],
            [
                [0, 8],
                [7, 0],
                [-1, 6],
            ],
            [
                [2, 4],
                [6, 8],
                [10, -12],
            ],
        ])
        b = xp.array([
            [
                [-2, 1, -1, 3],
            ],
            [
                [-5, 3, 0, 2],
            ]
        ])
        c = xp.array([
            [0, 2, -3, 0],
            [1, -1, 1, -1],
            [3, -5, 6, -2],
            [2, -3, -2, 3],
            [5, 9, 6, 3],
        ])

        self.assertEqual(xp.einsum('ijk,kli,mi->mjl', a, b, c).tolist(), [
            [
                [-18],
                [47],
                [-13],
            ],
            [
                [-17],
                [-80],
                [-40],
            ],
            [
                [-19],
                [-253],
                [-101],
            ],
            [
                [45],
                [25],
                [-49],
            ],
            [
                [ -99],
                [47],
                [-221],
            ],
        ])

    def test_getitem(self):
        a = xp.array([[[5]]])

        self.assertEqual(a[0][0][0], 5)
        self.assertEqual(a[0, 0, 0].tolist(), 5)
        self.assertEqual(a[:, :, :].tolist(), [[[5]]])

        a = xp.array([
            [1, 2],
            [3, 4]
        ])

        self.assertEqual(a[0, 0].tolist(), 1)
        self.assertEqual(a[0, 1].tolist(), 2)
        self.assertEqual(a[1, 0].tolist(), 3)
        self.assertEqual(a[1, 1].tolist(), 4)
        self.assertEqual(a[0, :].tolist(), [1, 2])
        self.assertEqual(a[1, :].tolist(), [3, 4])
        self.assertEqual(a[:, 0].tolist(), [1, 3])
        self.assertEqual(a[:, 1].tolist(), [2, 4])
        self.assertEqual(a[:, :].tolist(), [
            [1, 2],
            [3, 4]
        ])

        with self.assertRaises(IndexError):
            a[:, :, :]

        a = xp.array([
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ],
        ])

        self.assertEqual(a[0, 0, 0].tolist(), 1)
        self.assertEqual(a[0, 0, 1].tolist(), 2)
        self.assertEqual(a[0, 0, 2].tolist(), 3)
        self.assertEqual(a[0, 1, 0].tolist(), 4)
        self.assertEqual(a[0, 1, 1].tolist(), 5)
        self.assertEqual(a[0, 1, 2].tolist(), 6)
        self.assertEqual(a[1, 0, 0].tolist(), 7)
        self.assertEqual(a[1, 0, 1].tolist(), 8)
        self.assertEqual(a[1, 0, 2].tolist(), 9)
        self.assertEqual(a[1, 1, 0].tolist(), 10)
        self.assertEqual(a[1, 1, 1].tolist(), 11)
        self.assertEqual(a[1, 1, 2].tolist(), 12)

        with self.assertRaises(IndexError):
            a[1, 1, 3]

        self.assertEqual(a[:, :, :].tolist(), [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ])
        self.assertEqual(a[:, 1].tolist(), [
            [4, 5, 6],
            [10, 11, 12]
        ])
        self.assertEqual(a[:, :].tolist(), [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ])
        self.assertEqual(a[:, 1, 0].tolist(), [4, 10])
        self.assertEqual(a[1, 0, 2].tolist(), 9)

    def test_setitem(self):
        a = xp.array([[[5]]])

        a[0, 0, 0] = -1
        self.assertEqual(a[:, :, :].tolist(), [[[-1]]])

        a = xp.array([
            [1, 2],
            [3, 4]
        ])

        a[1, 1] = 5

        self.assertEqual(a[:, :].tolist(), [
            [1, 2],
            [3, 5]
        ])

        a[:, 1] = 2 * a[:, 1]

        self.assertEqual(a[:, :].tolist(), [
            [1, 4],
            [3, 10]
        ])

        a[:, :] = -2

        self.assertEqual(a[:, :].tolist(), [
            [-2, -2],
            [-2, -2]
        ])

        with self.assertRaises(IndexError):
            a[:, :, :] = 1

        a = xp.array([
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ])

        a[:, :, 1] = 0

        self.assertEqual(a[:, :].tolist(), [
            [
                [1, 0, 3],
                [4, 0, 6]
            ],
            [
                [7, 0, 9],
                [10, 0, 12]
            ]
        ])

        a = xp.array([
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ])

        a[1, :, 1:] = [
            [-1, -2],
            [-3, -4]
        ]

        self.assertEqual(a[:, :].tolist(), [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, -1, -2],
                [10, -3, -4]
            ],
        ])

        a = xp.array([
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ])

        a[0, :, :2] = [-2, -1]

        self.assertEqual(a[:, :].tolist(), [
            [
                [-2, -1, 3],
                [-2, -1, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ])

        with self.assertRaises(ValueError):
            a[0, :, :2] = [
                [-2, -1],
                [1, 1],
                [1, 1]
            ]

    def test_conj(self):
        a = xp.array(5).conj()

        self.assertEqual(a.tolist(), 5)

        a = xp.array(5.3).conj()

        self.assertEqual(a.tolist(), 5.3)

        a = xp.array(5 + 3*1j).conj()

        self.assertEqual(a.tolist(), 5 - 3*1j)

        a = xp.array([
            [1, 2, 3],
            [4, 5, 6],
        ]).conj()

        self.assertEqual(a.tolist(), [
            [1, 2, 3],
            [4, 5, 6],
        ])

        a = xp.array([
            [-1.2, 2.3, -3.4],
            [4.5, -5.6, 6.7],
        ]).conj()

        self.assertEqual(a.tolist(), [
            [-1.2, 2.3, -3.4],
            [4.5, -5.6, 6.7],
        ])

        a = xp.array([
            [-1.2 + 5.6*1j, 2.3 - 4.5*1j, -3.4 + 1.2*1j],
            [4.5 - 6.7*1j, -5.6 - 2.3*1j, 6.7 + 3.4*1j],
        ]).conj()

        self.assertEqual(a.tolist(), [
            [-1.2 - 5.6*1j, 2.3 + 4.5*1j, -3.4 - 1.2*1j],
            [4.5 + 6.7*1j, -5.6 + 2.3*1j, 6.7 - 3.4*1j],
        ])
