
"""Reduce the bond dimension of the Fibonacci MPO."""

import numpy as np
from .fibonacci import fibonacci_bulk
from dmrg.mpo import SimpleMPO


def _row_basis(rows, tol=1e-11):
    M = np.concatenate(rows, axis=0)
    _, s, vh = np.linalg.svd(M, full_matrices=False)
    return vh[:np.count_nonzero(s > tol * s[0])]


def fibonacci_small_bulk(jeff, penalty=75.0, theta=6*np.pi/5,
                         x_final=1, boundary_penalty=None):

    first, even, odd, last = fibonacci_bulk(
        jeff, penalty, theta, x_final, boundary_penalty
    )

    D = even.shape[0]

    before = [
        np.eye(D, dtype=complex)[:1],
        np.zeros((0, D), complex)
    ]

    for _ in range(2*D + 2):
        next_before = [before[0], before[1]]

        for parity, W in enumerate((even, odd)):
            rows = np.einsum(
                'ia,astb->istb', before[parity], W
            ).reshape(-1, D)

            dest = 1-parity

            next_before[dest] = _row_basis([
                before[dest], rows
            ])

        if all(a.shape[0] == b.shape[0]
               for a, b in zip(before, next_before)):
            before = next_before
            break

        before = next_before

    else:
        raise RuntimeError(
            'MPO reachable-space reduction did not converge'
        )

    B0, B1 = before

    reduced_even = np.einsum(
        'ia,astb,jb->istj',
        B0, even, B1.conj(),
        optimize=True
    )

    reduced_odd = np.einsum(
        'ia,astb,jb->istj',
        B1, odd, B0.conj(),
        optimize=True
    )

    reduced_first = np.einsum(
        'astb,jb->astj',
        first, B1.conj(),
        optimize=True
    )

    reduced_last = np.einsum(
        'ia,astb->istb',
        B1, last,
        optimize=True
    )

    return (
        reduced_first,
        reduced_even,
        reduced_odd,
        reduced_last
    )


def fibonacci_small_mpo(L, jeff, penalty=75.0,
                        theta=6*np.pi/5,
                        x_final=1, boundary_penalty=None):

    if L < 4 or L % 2:
        raise ValueError('L must be even and >= 4')

    first, even, odd, last = fibonacci_small_bulk(
        jeff, penalty, theta, x_final, boundary_penalty
    )

    return SimpleMPO(
        [first]
        + [even if j % 2 == 0 else odd
           for j in range(1, L-1)]
        + [last]
    )
