import random
import unittest

import mynumpy as mynp


class TestNdArray(unittest.TestCase):
    def setUp(self):
        ...

    def tearDown(self):
        ...

    def test_matrix_rank(self):
        a = mynp.array(
            [
                [1, 2],
                [3, 4],
            ]
        )
        self.assertEqual(mynp.linalg.matrix_rank(a), 2)

        a = mynp.array(
            [
                [1, 2],
                [2, 4],
            ]
        )
        self.assertEqual(mynp.linalg.matrix_rank(a), 1)

        a = mynp.array(
            [
                [0, 0],
                [0, 0],
            ]
        )
        self.assertEqual(mynp.linalg.matrix_rank(a), 0)

        a = mynp.array(
            [
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        self.assertEqual(mynp.linalg.matrix_rank(a), 0)

        a = mynp.array(
            [
                [1, 2, -3],
                [-4, 5, -6],
            ]
        )
        self.assertEqual(mynp.linalg.matrix_rank(a), 2)

    def test_real_svd(self):
        random.seed(1234)
        for _ in range(10):
            m = random.choice(list(range(1, 10)))
            n = random.choice(list(range(1, 10)))
            data = [(random.random() - 0.5) * 20 for _ in range(m * n)]
            a = mynp.array(data, dtype=float).reshape(m, n)
            U, S, Vh = mynp.linalg.svd(a)
            self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

    def test_complex_svd(self):
        random.seed(1234)
        for _ in range(10):
            m = random.choice(list(range(1, 10)))
            n = random.choice(list(range(1, 10)))
            data = [(random.random() - 0.5) * 20 for _ in range(m * n)]
            a = mynp.array(data, dtype=complex).reshape(m, n)
            a_real = a.real
            U, S, Vh = mynp.linalg.svd(a)
            self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a_real))
