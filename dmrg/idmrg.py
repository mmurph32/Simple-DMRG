"""Minimal centre-growing (infinite-system) two-site DMRG.

The algorithm grows an *open-chain superblock* by two sites per iteration.
It is a straightforward iDMRG growth algorithm, not a fully periodic iMPS
implementation with a fixed unit cell and transfer matrix.
"""

import numpy as np

from dmrg.mpo import SimpleMPO
from dmrg.mps import SimpleMPS
from dmrg.eigensolver import eigensolver
from dmrg.environments import update_left_env, update_right_env


def idmrg(
    mpo3,
    phi,
    chi_max,
    max_sites,
    svd_min=1e-12,
    tol=None,
    stable_steps=3,
    verbose=True
):
    """Grow a translation-invariant open-chain MPO from length 2 to max_sites.

    Parameters
    ----------
    mpo3: SimpleMPO
        MPO for a 3-site open chain: tensors [W_first, W_bulk, W_last].
    phi: array_like
        Normalized local initial guess, e.g. [1, 0] for spin-1/2.
    chi_max: int
        Maximum MPS virtual bond dimension.
    max_sites: int
        Even chain length at which to stop (two sites added per iteration).
    svd_min: float
        Relative singular-value cutoff for SVD truncation.
    verbose: bool
        Print current size and estimated bulk energy density.

    Returns
    -------
    mps: SimpleMPS
        Final growing superblock reconstructed as a finite open-chain MPS.
    history: list[dict]
        Number of sites, total energy after truncation, bulk energy estimate,
        and retained central bond dimension at each growth step.
    """
    if max_sites < 2 or max_sites % 2:
        raise ValueError("max_sites must be an even integer >= 2")
    if chi_max < 1:
        raise ValueError("chi_max must be positive")
    if len(mpo3.mpo_tensors) != 3:
        raise ValueError("mpo3 must have first, bulk, and last MPO tensors")

    W_first, W_bulk, W_last = mpo3.mpo_tensors
    dtype = np.result_type(np.asarray(phi).dtype, W_bulk.dtype, np.float64)
    phi = np.asarray(phi, dtype=dtype).reshape(-1)
    norm = np.linalg.norm(phi)
    if norm == 0:
        raise ValueError("phi must be nonzero")
    phi = phi / norm
    d = phi.size

    left_env = np.ones((1, 1, 1), dtype=dtype)
    right_env = np.ones((1, 1, 1), dtype=dtype)
    schmidt = np.ones(1, dtype=np.float64)

    # These two MPOs are local windows, NOT standalone MPOs for the grown chain.
    first_window = SimpleMPO([W_first, W_last])
    bulk_window = SimpleMPO([W_bulk, W_bulk])

    left_sites = []       # left boundary -> centre
    right_sites = []      # centre -> right boundary
    history = []
    previous_energy = None

    stable_count = 0

    for step in range(max_sites // 2):
        if step == 0:
            W_l, W_r, window = W_first, W_last, first_window
        else:
            W_l, W_r, window = W_bulk, W_bulk, bulk_window

        chi_old = len(schmidt)
        # A current Schmidt bond plus two newly inserted physical sites:
        # theta[a,s,t,b] = schmidt[a] * delta[a,b] * phi[s] * phi[t].
        theta = np.einsum(
            'ab,s,t->astb', np.diag(schmidt), phi, phi
        )

        _, optimized = eigensolver(
            None, window, 0, theta, left_env, right_env
        )

        # SVD separating the two sites; KEEP S outside A and B so both
        # tensors are isometries for the next environment update.
        U, s, Vh = np.linalg.svd(
            optimized.reshape(chi_old * d, d * chi_old),
            full_matrices=False,
        )
        s_norm = np.linalg.norm(s)
        if s_norm == 0:
            raise RuntimeError("Local eigensolver returned a zero vector")
        chi_new = min(chi_max, max(1, int(np.count_nonzero(s / s_norm >= svd_min))))
        discarded_weight = float(np.sum((s[chi_new:] / s_norm) ** 2))
        schmidt = s[:chi_new] / np.linalg.norm(s[:chi_new])

        A = U[:, :chi_new].reshape(chi_old, d, chi_new)
        B = Vh[:chi_new, :].reshape(chi_new, d, chi_old)

        left_env = update_left_env(left_env, A, W_l)
        right_env = update_right_env(right_env, B, W_r)

        left_sites.append(A)
        right_sites.insert(0, B)

        # Evaluate energy of the *truncated*, normalized state, not just the
        # energy of the pre-truncation two-site eigenvector.
        E = np.einsum(
            'awb,awb,a,b->', left_env, right_env, schmidt, schmidt
        ).real.item()
        L = 2 * (step + 1)
        # Calculate bulk energy
        e_bulk = None if previous_energy is None else (E - previous_energy) / 2

        history.append(dict(
            L=L,
            E=E,
            E_per_site=E / L,
            e_bulk=e_bulk,
            chi=chi_new,
            discarded_weight=discarded_weight
        ))

        previous_energy = E

        # Print current iteration 
        if verbose and (L % 10 == 0 or L == 2 or L == max_sites):
            print(
                f"Growth: L={L:4d}/{max_sites} "
                f"chi={chi_new:4d} "
                f"E={E:.10f}",
                flush=True
            )

        # Check convergence
        if tol is not None and len(history) >= 2:

            previous_bulk = history[-2]["e_bulk"]

            if previous_bulk is not None:

                delta_e = abs(e_bulk - previous_bulk)

                if delta_e < tol:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= stable_steps:
                    if verbose:
                        print(f"iDMRG converged at L={L}")
                    break

    # Reconstruct the latest open-chain state: left ... A diag(s) B ... right.
    # Only the final left-isometric tensor absorbs the Schmidt coefficients.
    final_left = left_sites[:-1] + [left_sites[-1] * schmidt[None, None, :]]
    mps = SimpleMPS(final_left + right_sites)
    return mps, history
