
"""Centre-growing Fibonacci DMRG, followed by finite sweeps.

Supports even and odd system sizes.
For odd L, grow to L-1 and insert one site before the right boundary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from AGS_Anyon_Model.compressed import (
    fibonacci_small_bulk as fibonacci_bulk,
    fibonacci_small_mpo as fibonacci_scalable_mpo
)

from dmrg.mpo import SimpleMPO
from dmrg.mps import SimpleMPS
from dmrg.eigensolver import eigensolver, right_canonicalize
from dmrg.environments import update_left_env, update_right_env
from dmrg.sweep import sweep
from dmrg.energy import energy


def anyon_idmrg(jeff, chi_max, max_sites, penalty=75.0,
                svd_min=1e-12, verbose=True, x_final=1,
                boundary_penalty=None):

    """Grow by two sites. For odd L, initialize one extra site."""

    if max_sites < 4:
        raise ValueError("max_sites must be >= 4")

    # Centre growth is performed to an even length
    grow_sites = max_sites - (max_sites % 2)

    first, even, odd, last = fibonacci_bulk(
        jeff, penalty,
        x_final=x_final,
        boundary_penalty=boundary_penalty
    )

    left_env = np.ones((1, 1, 1), dtype=complex)
    right_env = np.ones((1, 1, 1), dtype=complex)

    schmidt = np.ones(1)
    phi = np.array([1, 1], dtype=complex) / np.sqrt(2)

    left_sites = []
    right_sites = []
    history = []
    previous_energy = None

    for step in range(grow_sites // 2):

        L = 2 * (step + 1)

        if step == 0:
            W_l, W_r = first, last
        else:
            j = L // 2 - 1
            W_l = even if j % 2 == 0 else odd
            W_r = even if (j+1) % 2 == 0 else odd

        window = SimpleMPO([W_l, W_r])

        chi_old = len(schmidt)

        theta = np.einsum(
            'ab,s,t->astb',
            np.diag(schmidt), phi, phi
        )

        _, optimized = eigensolver(
            None, window, 0, theta,
            left_env, right_env
        )

        U, s, Vh = np.linalg.svd(
            optimized.reshape(chi_old*2, 2*chi_old),
            full_matrices=False
        )

        s /= np.linalg.norm(s)

        chi_new = min(
            chi_max,
            max(1, np.count_nonzero(s >= svd_min))
        )

        discarded = float(np.sum(s[chi_new:]**2))

        schmidt = s[:chi_new] / np.linalg.norm(s[:chi_new])

        A = U[:, :chi_new].reshape(chi_old, 2, chi_new)
        B = Vh[:chi_new].reshape(chi_new, 2, chi_old)

        left_env = update_left_env(left_env, A, W_l)
        right_env = update_right_env(right_env, B, W_r)

        left_sites.append(A)
        right_sites.insert(0, B)

        E = np.einsum(
            'awb,awb,a,b->',
            left_env, right_env,
            schmidt, schmidt
        ).real.item()

        history.append(dict(
            L=L,
            E=E,
            chi=chi_new,
            e_bulk=None if previous_energy is None
            else (E-previous_energy)/2,
            discarded_weight=discarded
        ))

        previous_energy = E

        if verbose and (L % 10 == 0 or L == grow_sites):
            print(
                f"Growth: L={L:4d} chi={chi_new:4d} E={E:.10f}",
                flush=True
            )

    left_sites[-1] *= schmidt[None, None, :]

    tensors = left_sites + right_sites

    # Odd-length initialization
    if max_sites % 2:

        # Insert a new site before the final boundary tensor.
        # A product-state insertion preserves compatible MPS bonds.
        last_tensor = tensors[-1]
        chi_left = last_tensor.shape[0]

        new_site = np.zeros(
            (chi_left, 2, chi_left),
            dtype=complex
        )

        # Initialize the new fusion label to tau
        new_site[:, 1, :] = np.eye(chi_left)

        tensors.insert(-1, new_site)

        history.append(dict(
            L=max_sites,
            E=None,
            chi=chi_left,
            e_bulk=None,
            discarded_weight=0.0
        ))

        if verbose:
            print(f"Inserted extra site: L={max_sites}")

    return SimpleMPS(tensors), history


def anyon_grow_and_sweep(jeff, chi_max, max_sites, penalty=75.0,
                         number_of_sweeps=10, tol=1e-9,
                         verbose=True, x_final=1,
                         boundary_penalty=None):

    mpo = fibonacci_scalable_mpo(
        max_sites, jeff, penalty,
        x_final=x_final,
        boundary_penalty=boundary_penalty
    )

    mps, history = anyon_idmrg(
        jeff, chi_max, max_sites, penalty,
        verbose=verbose,
        x_final=x_final,
        boundary_penalty=boundary_penalty
    )

    mps = right_canonicalize(mps)

    E_before = energy(mps, mpo)

    mps, E_after = sweep(
        mps, mpo,
        bond_dim=chi_max,
        number_of_sweeps=number_of_sweeps,
        tol=tol,
        verbose=verbose
    )

    return mps, E_after, E_before, history
