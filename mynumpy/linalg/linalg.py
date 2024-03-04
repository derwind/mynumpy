from __future__ import annotations

from .. import array, eye, ndarray, sqrt


def matrix_rank(G: ndarray, tol: float | None = None) -> int:
    if tol is None:
        tol = 1e-8

    if len(G.shape) != 2:
        raise ValueError("expected 2-dimensional array")

    if G.dtype == complex:
        if G._is_true_complex():
            raise NotImplementedError("true complex matrices are not supported")

    if G.shape[0] < G.shape[1]:
        G = G.T

    if G.dtype == int:
        U = G.astype(float)
    elif G.dtype == complex:
        U = G.real
    else:
        U = G.copy()

    _, n = U.shape

    try:
        _update_UV(U, tol=tol)
    except OverflowError:
        G.data[0][0] += tol  # W/A for non-invertible matrix
        return matrix_rank(G, tol)

    S = []
    for i in range(n):
        u_i = U[:, i]
        sigma = sqrt(sum(u_i * u_i))
        if sigma < tol:
            continue
        S.append(sigma)

    return len(S)


def svd(G: ndarray, *args) -> tuple[ndarray, ndarray, ndarray]:
    """Singular Value Decomposition

    Args
        G (ndarray): The input matrix. Must be 2-dimensional.

    Returns:
        ndarray: The left singular vectors.
        ndarray: The singular values.
        ndarray: The right singular vectors.
    """

    tol = 1e-8

    if len(G.shape) != 2:
        raise ValueError("expected 2-dimensional array")

    is_complex = False
    if G.dtype == complex:
        if G._is_true_complex():
            raise NotImplementedError("true complex matrices are not supported")
        is_complex = True

    need_reverse = False
    if G.shape[0] < G.shape[1]:
        G = G.T
        need_reverse = True

    if G.dtype == int:
        U = G.astype(float)
    elif G.dtype == complex:
        U = G.real
    else:
        U = G.copy()

    _, n = U.shape

    S = []
    V = eye(n)

    try:
        _update_UV(U, V, tol)
    except OverflowError:
        G.data[0][0] += tol  # W/A for non-invertible matrix
        return svd(G, *args)

    for i in range(n):
        u_i = U[:, i]
        sigma = sqrt(sum(u_i * u_i))
        S.append(sigma)
        U[:, i] = U[:, i] / sigma

    if need_reverse:
        U, V = V, U

    orig_S = S[:]
    S.sort(reverse=True)
    order = [orig_S.index(v) for v in S]
    if is_complex:
        U = U.astype(complex)
        V = V.astype(complex)
        S = array(S, dtype=complex)
    else:
        S = array(S, dtype=float)
    Uh = U.T
    Vh = V.T
    Uh.data = [Uh.data[i] for i in order]
    Vh.data = [Vh.data[i] for i in order]

    return Uh.T, S, Vh


def _update_UV(U: ndarray, V: ndarray | None = None, tol: float = 1e-8) -> None:
    _, n = U.shape
    while True:
        OKs = []
        for i in range(n - 1):
            u_i = U[:, i]
            v_i = V[:, i] if V is not None else None
            for j in range(i + 1, n):
                u_j = U[:, j]
                a = sum(u_i * u_i)
                b = sum(u_j * u_j)
                c = sum(u_i * u_j)
                OKs.append(c**2 <= tol**2 * a * b)
                if c == 0:
                    continue

                zeta = (b - a) / c
                t = 1 / (abs(zeta) + sqrt(1 + zeta**2))
                if zeta < 0:
                    t *= -1
                cs = 1 / sqrt(1 + t**2)
                sn = cs * t
                RT = array([[cs, -sn], [sn, cs]])
                tmp = RT @ array([u_i.data, u_j.data])
                u_i = tmp[0]
                u_j = tmp[1]
                U[:, i] = u_i
                U[:, j] = u_j
                u_i = array(u_i)
                u_j = array(u_j)

                if V is not None:
                    v_j = V[:, j]
                    tmp = RT @ array([v_i.data, v_j.data])
                    v_i = tmp[0]
                    v_j = tmp[1]
                    V[:, i] = v_i
                    V[:, j] = v_j
                    v_i = array(v_i)
                    v_j = array(v_j)

        try_again = False
        for ok in OKs:
            if not ok:
                try_again = True
                break
        if not try_again:
            break
