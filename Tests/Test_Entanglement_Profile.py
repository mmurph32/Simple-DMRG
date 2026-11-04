
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from dmrg.mps import product_state
from dmrg.tfim import tfim_mpo
from dmrg.sweep import sweep
from dmrg.entanglement import entanglement_entropy
from dmrg.central_charge import fit_central_charge


# Parameters
L = 200
J = 1.0
h = 1.0
bond_dim = 64

fit_trim = 20

# Run DMRG
mps = product_state([1.0, 0.0], L)
mpo = tfim_mpo(L, J, h)

final_mps, ground_energy = sweep(
    mps, mpo,
    bond_dim=bond_dim,
    number_of_sweeps=20,
    tol=1e-10
)

# Calculate entanglement entropy
S_dmrg = entanglement_entropy(final_mps)

# Calabrese-Cardy fit
c, slope, intercept, x = fit_central_charge(
    S_dmrg, trim=fit_trim
)

relative_error = abs(c - 0.5) / 0.5 * 100

print(f"Ground energy: {ground_energy:.12f}")
print(f"Central charge: c = {c:.6f}")
print("Expected Ising: c = 0.500000")
print(f"Relative error: {relative_error:.3f}%")

ell = np.arange(1, L)
S_fit = slope * x + intercept


# Plot 1: Entropy vs bond position
plt.figure()
plt.scatter(ell, S_dmrg, s=15, label="DMRG")
plt.plot(ell, S_fit, "r-", label=f"CC fit: c = {c:.4f}")
plt.xlabel("Bond position")
plt.ylabel("Entanglement entropy")
plt.legend()
plt.show()


# Plot 2: Calabrese-Cardy linear fit
plt.figure()
plt.scatter(x, S_dmrg, s=15, label="DMRG")
plt.plot(x, S_fit, "r-", label=f"Fit: c = {c:.4f}")
plt.xlabel(r"$\ln\left[\frac{2L}{\pi}\sin\left(\frac{\pi\ell}{L}\right)\right]$")
plt.ylabel(r"$S(\ell)$")
plt.legend()
plt.show()


# Check stability of fitted central charge
trims = np.arange(0, 51, 2)
c_values = []

for trim in trims:
    c_test, _, _, _ = fit_central_charge(
        S_dmrg, trim=trim
    )
    c_values.append(c_test)

plt.figure()
plt.plot(trims, c_values, "o-")
plt.axhline(0.5, color="red", linestyle="--",
            label="Exact Ising c = 0.5")
plt.xlabel("Boundary trim")
plt.ylabel("Fitted central charge")
plt.legend()
plt.show()