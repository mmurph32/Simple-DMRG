import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from dmrg.mps import product_state
from dmrg.tfim import tfim_mpo
from dmrg.sweep import sweep
import matplotlib.pyplot as plt
from dmrg.entanglement import entanglement_entropy
from dmrg.central_charge import extract_central_charge

L = 100
J = 1.0
h = 1.0
bond_dim = 12  # for full capture of an arbitrary state, need d^(L/2) 

mps = product_state([1.0, 0.0], L)
mpo = tfim_mpo(L, J, h)

final_mps, ground_energy = sweep(
    mps, mpo,
    bond_dim=bond_dim,
    number_of_sweeps=20,
    tol=1e-10
)


# Calculate DMRG entropies
S_dmrg = entanglement_entropy(final_mps)

# Plot comparison
ell_values = np.arange(1, L)

# Extract Central Charge

c_dmrg = extract_central_charge(S_dmrg)[0]
#c_exact = extract_central_charge(S_exact)[0]

print("DMRG Central Charge c =", c_dmrg)
#print("Exact Central Charge c =", c_exact)

#%%


# %% Plot Calabrese-Cardy Fit

c, slope, intercept = extract_central_charge(S_dmrg)

L_range = np.arange(1, L)

# Conformal distance
x = np.log(
    (2 * L / np.pi) * np.sin(np.pi * L_range / L)
)

# Predicted entropy from fitted parameters
S_fit = slope * x + intercept

plt.figure()
plt.plot(L_range, S_dmrg, "o", label="DMRG")
plt.plot(L_range, S_fit, "-", label=f"CC fit: c = {c:.4f}")

plt.xlabel("Bond position")
plt.ylabel("Entanglement entropy")
plt.legend()
plt.show()


#%%

# %% Calabrese-Cardy Linear Fit

c, slope, intercept = extract_central_charge(S_dmrg)

L_range = np.arange(1, L)

x = np.log(
    (2 * L / np.pi) * np.sin(np.pi * L_range / L)
)

plt.figure()

plt.scatter(x, S_dmrg, label="DMRG", color="blue")
plt.plot(x, slope*x + intercept, "r-", label=f"Fit: c = {c:.4f}")

plt.xlabel(r"$\ln\left[\frac{2L}{\pi}\sin\left(\frac{\pi\ell}{L}\right)\right]$")
plt.ylabel(r"$S(\ell)$")
plt.legend()
plt.show()

#%%
