
import numpy as np


def bond_correlations(mps):
    """
    B(ell) = <n_(ell-1) n_(ell+1)>
    for cuts ell = 1, ..., L-2.

    Returns an array of length L-1.
    The final entry is NaN (no corresponding operator).
    """
    A = mps.tensors
    L = mps.L
    n = np.diag([0., 1.])

    left = [None] * (L + 1)
    right = [None] * (L + 1)

    left[0] = np.ones((1, 1), dtype=complex)
    right[L] = np.ones((1, 1), dtype=complex)

    def transfer(env, T, op):
        return np.einsum(
            "ab,asr,st,btu->ru",
            env, T.conj(), op, T,
            optimize=True
        )

    def right_transfer(env, T):
        return np.einsum(
            "asr,bsu,ru->ab",
            T.conj(), T, env,
            optimize=True
        )

    I = np.eye(2)

    for i in range(L):
        left[i + 1] = transfer(left[i], A[i], I)

    for i in range(L - 1, -1, -1):
        right[i] = right_transfer(right[i + 1], A[i])

    norm = left[L][0, 0].real
    B = np.full(L - 1, np.nan)

    for ell in range(1, L - 1):
        i = ell - 1

        env = transfer(left[i], A[i], n)
        env = transfer(env, A[i + 1], I)
        env = transfer(env, A[i + 2], n)

        B[ell - 1] = (
            np.einsum("ab,ab->", env, right[i + 3]).real
            / norm
        )

    return B


def fit_reduced_entropy(S, B, trim=5):
    """
    Fit S = (c/6)*x + intercept + zeta*B.
    Open boundary conditions, natural logarithms.
    """
    S = np.asarray(S)
    B = np.asarray(B)

    L = len(S) + 1
    ell = np.arange(1, L)

    x = np.log(
        (2 * L / np.pi) * np.sin(np.pi * ell / L)
    )

    mask = (
        (ell > trim)
        & (ell < L - trim)
        & np.isfinite(S)
        & np.isfinite(B)
    )

    X = np.column_stack([
        x[mask],
        np.ones(mask.sum()),
        B[mask]
    ])

    slope, intercept, zeta = np.linalg.lstsq(
        X, S[mask], rcond=None
    )[0]

    c = 6 * slope

    S_reduced = S - zeta * B
    S_fit = slope * x + intercept

    return c, zeta, S_reduced, S_fit, x
