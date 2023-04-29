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

        self.assertEqual(a.reshape((1, 4)).data, [1, 2, 3, 4])
        self.assertEqual(a.reshape(1, 4).data, [1, 2, 3, 4])
        self.assertEqual(a.reshape((-1, 4)).data, [1, 2, 3, 4])
        self.assertEqual(a.reshape(-1, 4).data, [1, 2, 3, 4])
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
