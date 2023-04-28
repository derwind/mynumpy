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
