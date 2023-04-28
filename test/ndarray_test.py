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
