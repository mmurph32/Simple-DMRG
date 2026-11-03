import numpy as np
from scipy.sparse.linalg import eigsh

from dmrg.constrained_mps import constrained_random_mps
from dmrg.constrained_mpo import constrained_hard_boson_mpo
from dmrg.constrained_hard_boson import constrained_hamiltonian
from dmrg.constrained_sweep import constrained_sweep
from dmrg.entanglement import entanglement_entropy

L = 97
w = 1.0
U = -3.0299
V = 3.3302

chi = 64

# Constrained DMRG
mpo = constrained_hard_boson_mpo(L, w, U, V)
mps = constrained_random_mps(L)

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

mps, E_dmrg = constrained_sweep(
    mps, mpo,
    bond_dim=chi,
    number_of_sweeps=10,
    tol=1e-7
)

profiler.disable()

pstats.Stats(profiler).sort_stats(
    "cumulative"
).print_stats(15)

S = entanglement_entropy(mps)

from dmrg.reduced_entropy import (
    bond_correlations,
    fit_reduced_entropy
)

B = bond_correlations(mps)

np.savez(
    "hard_boson_observables.npz",
    S=S,
    B=B
)

c, zeta, S_reduced, S_fit, x = fit_reduced_entropy(
    S, B, trim=8
)

print(f"Reduced central charge: c = {c:.6f}")
print(f"Correction coefficient: zeta = {zeta:.6f}")