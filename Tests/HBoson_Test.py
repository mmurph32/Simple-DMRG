import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import time
from dmrg.hard_boson import hard_boson_mpo
from dmrg.grow_and_sweep import grow_and_sweep
from dmrg.entanglement import entanglement_entropy
from dmrg.central_charge import fit_central_charge


# Parameters: approximate Potts critical point
L = 200
w = 1.0
U = -1.308
V = 0.0
penalty = 50.0

chi = 64
tol = 1e-10


# Build MPOs
mpo3 = hard_boson_mpo(3, w, U, V, penalty)
mpo = hard_boson_mpo(L, w, U, V, penalty)


# Grow and polish
start = time.perf_counter()

mps, E, E_before, history = grow_and_sweep(
    mpo3=mpo3,
    mpo=mpo,
    phi=[1.0, 0.0],
    chi_max=chi,
    number_of_sweeps=20,
    tol=tol,
    verbose=True
)

runtime = time.perf_counter() - start


# Entanglement entropy
S = entanglement_entropy(mps)

np.save("hard_boson_entropy.npy", S)

print("Entropy saved.")