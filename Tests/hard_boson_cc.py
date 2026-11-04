import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dmrg.hard_boson import hard_boson_mpo
from dmrg.central_charge import central_charge_convergence


# Hard-boson Ising transition
w = 1.0
U = -5.8902
V = 2.0
penalty = 100.0

results = central_charge_convergence(
    mpo_builder=lambda L: hard_boson_mpo(
        L, w, U, V, penalty
    ),
    L_values=[50, 100, 200],
    chi_values=[2, 4, 8],
    trim_max=0.4,
    trim_steps=21,
    sweeps=20,
    tol=1e-10,
    expected_c=0.5
)

np.savez("hard_boson_c_results.npz", results)

print("Results saved.")
