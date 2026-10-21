import numpy as np
from dmrg.mps import product_state
from dmrg.tfim import tfim_mpo
from dmrg.sweep import sweep
import matplotlib.pyplot as plt
from dmrg.entanglement import entanglement_entropy
from dmrg.central_charge import extract_central_charge

L = 12
J = 1.0
h = 1.0
bond_dim = 4  # for full capture of an arbitrary state, need d^(L/2) 

mps = product_state([1.0, 0.0], L)
mpo = tfim_mpo(L, J, h)

final_mps, ground_energy = sweep(
    mps, mpo,
    bond_dim=bond_dim,
    number_of_sweeps=20,
    tol=1e-16
)

# Run this after DMRG has converged
# %% Compare entanglement with exact diagonalisation

import numpy as np
import matplotlib.pyplot as plt

# Construct and diagonalise the full Hamiltonian
H = mpo.contract()
eigenvalues, eigenvectors = np.linalg.eigh(H)

exact_energy = eigenvalues[0]
psi_exact = eigenvectors[:, 0]

# Calculate exact entanglement entropy
S_exact = []

for ell in range(1, L):

    # Split Hilbert space into left and right subsystems
    psi_matrix = psi_exact.reshape(2**ell, 2**(L-ell))

    # Schmidt coefficients
    lambdas = np.linalg.svd(
        psi_matrix, compute_uv=False
    )

    p = lambdas**2
    p = p[p > 1e-15]

    entropy = -np.sum(p * np.log(p))
    S_exact.append(entropy)

# Calculate DMRG entropies
S_dmrg = entanglement_entropy(final_mps)

# Plot comparison
ell_values = np.arange(1, L)

plt.plot(ell_values, S_exact, "o-", label="Exact")
plt.plot(ell_values, S_dmrg, "x--", label="DMRG")

plt.xlabel("Bond position")
plt.ylabel("Entanglement entropy")
plt.legend()
plt.show()
#%%


# Extract Central Charge

c_dmrg = extract_central_charge(S_dmrg)[0]
c_exact = extract_central_charge(S_exact)[0]

print("DMRG Central Charge c =", c_dmrg)
print("Exact Central Charge c =", c_exact)

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
