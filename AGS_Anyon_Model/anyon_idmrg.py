"""Centre-growing Fibonacci DMRG, followed by finite sweeps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from AGS_Anyon_Model.fibonacci import fibonacci_bulk, fibonacci_scalable_mpo
from dmrg.mpo import SimpleMPO
from dmrg.mps import SimpleMPS
from dmrg.eigensolver import eigensolver, right_canonicalize
from dmrg.environments import update_left_env, update_right_env
from dmrg.sweep import sweep
from dmrg.energy import energy


def anyon_idmrg(jeff, chi_max, max_sites, penalty=75.0,
                svd_min=1e-12, verbose=True):
    """Grow by two fusion labels, keeping the alternating MPO parity."""
    if max_sites < 4 or max_sites % 2:
        raise ValueError('max_sites must be even and >= 4')

    first, even, odd, last = fibonacci_bulk(jeff, penalty)
    left_env = np.ones((1, 1, 1), dtype=complex)
    right_env = np.ones((1, 1, 1), dtype=complex)
    schmidt = np.ones(1)
    phi = np.array([1, 1], dtype=complex) / np.sqrt(2)  # tau labels, allowed initial state
    left_sites, right_sites, history = [], [], []
    previous_energy = None

    for step in range(max_sites // 2):
        L = 2 * (step + 1)
        if step == 0:
            W_l, W_r = first, last
        else:
            # New sites have indices L//2-1 and L//2.
            j = L // 2 - 1
            W_l = even if j % 2 == 0 else odd
            W_r = even if (j+1) % 2 == 0 else odd

        window = SimpleMPO([W_l, W_r])
        chi_old = len(schmidt)
        theta = np.einsum('ab,s,t->astb', np.diag(schmidt), phi, phi)
        _, optimized = eigensolver(None, window, 0, theta, left_env, right_env)

        U, s, Vh = np.linalg.svd(
            optimized.reshape(chi_old * 2, 2 * chi_old), full_matrices=False
        )
        s /= np.linalg.norm(s)
        chi_new = min(chi_max, max(1, np.count_nonzero(s >= svd_min)))
        discarded = float(np.sum(s[chi_new:]**2))
        schmidt = s[:chi_new] / np.linalg.norm(s[:chi_new])
        A = U[:, :chi_new].reshape(chi_old, 2, chi_new)
        B = Vh[:chi_new].reshape(chi_new, 2, chi_old)

        left_env = update_left_env(left_env, A, W_l)
        right_env = update_right_env(right_env, B, W_r)
        left_sites.append(A)
        right_sites.insert(0, B)

        E = np.einsum('awb,awb,a,b->', left_env, right_env,
                      schmidt, schmidt).real.item()
        history.append(dict(L=L, E=E, chi=chi_new,
                            e_bulk=None if previous_energy is None
                            else (E-previous_energy)/2,
                            discarded_weight=discarded))
        previous_energy = E
        if verbose and (L % 10 == 0 or L == max_sites):
            print(f'Growth: L={L:4d} chi={chi_new:4d} E={E:.10f}', flush=True)

    left_sites[-1] = left_sites[-1] * schmidt[None, None, :]
    return SimpleMPS(left_sites + right_sites), history


def anyon_grow_and_sweep(jeff, chi_max, max_sites, penalty=75.0,
                         number_of_sweeps=10, tol=1e-9, verbose=True):
    mpo = fibonacci_scalable_mpo(max_sites, jeff, penalty)
    mps, history = anyon_idmrg(jeff, chi_max, max_sites, penalty,
                              verbose=verbose)
    mps = right_canonicalize(mps)
    E_before = energy(mps, mpo)
    mps, E_after = sweep(mps, mpo, bond_dim=chi_max,
                         number_of_sweeps=number_of_sweeps,
                         tol=tol, verbose=verbose)
    return mps, E_after, E_before, history
