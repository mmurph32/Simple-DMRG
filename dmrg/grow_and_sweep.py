from dmrg.idmrg import idmrg
from dmrg.sweep import sweep
from dmrg.energy import energy
from dmrg.eigensolver import right_canonicalize



import time

from dmrg.idmrg import idmrg
from dmrg.sweep import sweep
from dmrg.energy import energy
from dmrg.eigensolver import right_canonicalize


def grow_and_sweep(
    mpo3,
    mpo,
    phi,
    chi_max,
    number_of_sweeps=20,
    tol=1e-10,
    svd_min=1e-12,
    verbose=True
):

    start_total = time.perf_counter()

    # Stage 1: Growing DMRG
    if verbose:
        print("\nStage 1: Growing DMRG", flush=True)

    start = time.perf_counter()

    mps, history = idmrg(
        mpo3=mpo3,
        phi=phi,
        chi_max=chi_max,
        max_sites=mpo.L,
        svd_min=svd_min,
        tol=None,
        verbose=verbose
    )

    if verbose:
        print(f"Growth completed in {time.perf_counter()-start:.2f} s",
              flush=True)

    # Stage 2: Canonicalisation
    if verbose:
        print("\nStage 2: Right canonicalisation", flush=True)

    start = time.perf_counter()

    mps = right_canonicalize(mps)

    E_before = energy(mps, mpo)

    if verbose:
        print(f"Canonicalisation completed in {time.perf_counter()-start:.2f} s",
              flush=True)
        print(f"Energy before polishing: {E_before:.12f}", flush=True)

    # Stage 3: Finite DMRG
    if verbose:
        print("\nStage 3: Finite DMRG polishing", flush=True)

    start = time.perf_counter()

    mps, E_after = sweep(
        mps,
        mpo,
        bond_dim=chi_max,
        number_of_sweeps=number_of_sweeps,
        tol=tol,
        verbose=verbose
    )

    if verbose:
        print(f"Polishing completed in {time.perf_counter()-start:.2f} s",
              flush=True)
        print(f"Final energy: {E_after:.12f}", flush=True)
        print(f"Total runtime: {time.perf_counter()-start_total:.2f} s",
              flush=True)

    return mps, E_after, E_before, history
