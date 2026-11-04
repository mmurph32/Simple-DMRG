import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dmrg.tfim import tfim_mpo
from dmrg.central_charge import central_charge_convergence

results = central_charge_convergence(
    mpo_builder=lambda L: tfim_mpo(L, J=1.0, h=1.0),
    L_values=[50, 100, 150, 200, 300],
    chi_values=[24, 48, 64],
    trim_max=0.4,
    trim_steps=21,
    sweeps=20,
    tol=1e-10,
    expected_c=0.5
)

np.savez("central_charge_results.npz", results)

print("Results saved.")