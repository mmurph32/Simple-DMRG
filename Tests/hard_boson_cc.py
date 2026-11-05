import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dmrg.hard_boson import hard_boson_mpo
from dmrg.central_charge import central_charge_convergence_grow_and_save


# Hard-boson Ising transition
w = 1.0
U = -1.308
V = 0.0
penalty = 50.0

results = central_charge_convergence_grow_and_save(
    mpo_builder=lambda L: hard_boson_mpo(
        L, w, U, V, penalty
    ),
    L_values=[48, 98, 148, 198, 298],
    chi_values=[12],
    trim_max=0.0,
    trim_steps=1,
    sweeps=20,
    tol=1e-10,
    expected_c=0.5
)

np.savez("hard_boson_c_results.npz", results)

print("Results saved.")
