import math
import random
import unittest

import mynumpy as mynp


class TestNdArray(unittest.TestCase):
    def setUp(self):
        ...

    def tearDown(self):
        ...

    def test_norm(self):
        a = mynp.array([1, 2])
        self.assertAlmostEqual(mynp.linalg.norm(a), math.sqrt(5))
        self.assertAlmostEqual(mynp.linalg.norm(a, ord=1), 3)

        a = mynp.array(
            [
                [1, 2],
                [3, 4],
            ]
        )
        self.assertAlmostEqual(mynp.linalg.norm(a), math.sqrt(30))
        self.assertAlmostEqual(mynp.linalg.norm(a, ord=1), 10)

        a = mynp.array(
            [
                [1 + 2j, -2 - 3j],
                [-3 + 1j, 4 + 5j],
            ]
        )
        self.assertAlmostEqual(mynp.linalg.norm(a), math.sqrt(69))
        self.assertAlmostEqual(mynp.linalg.norm(a, ord=1), 15.407021150565008)

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

    def test_svd(self):
        # identity
        data = [
            [1, 0],
            [0, 1],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        data = [
            [1 / math.sqrt(2), 0],
            [0, 1 / math.sqrt(2)],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        data = [
            [1, 2],
            [0, 4],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        # non-regular, rank = 1
        data = [
            [1, 2],
            [2, 4],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        data = [
            [1, 0],
            [0, 0],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        # zero
        data = [
            [0, 0],
            [0, 0],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        data = [
            [1, -2, 3],
            [-4, 5, -6],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        data = [
            [1, -2],
            [-3, 4],
            [5, -6],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        # Jordan block
        data = [
            [3, 1, 0],
            [0, 3, 1],
            [0, 0, 3],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        # non-regular
        data = [
            [1, 1, -2],
            [1, -2, 1],
            [-2, 1, 1],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        data = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        data = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        data = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

        # identity
        data = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
        a = mynp.array(data, dtype=float)
        U, S, Vh = mynp.linalg.svd(a)
        self.assertTrue(mynp.allclose(U @ mynp.diag(S) @ Vh, a))

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
