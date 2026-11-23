import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from dmrg.central_charge import fit_central_charge

# Load entropy
S = np.load("anyon_entropy_L=200_chi=64.npy")
L = len(S) + 1

# Calabrese-Cardy fit
trim = 40
c, slope, intercept, x = fit_central_charge(S, trim=trim)

L = len(S) + 1
ell = np.arange(1, L)

# Identify points used in the fit
mask = (ell > trim) & (ell < L - trim)

S_fit = slope * x + intercept

print(f"Central charge: c = {c:.6f}")

# Plot 1: Entropy vs bond position
plt.figure()

plt.scatter(ell[mask], S[mask], s=15, label="Used in fit")
plt.scatter(ell[~mask], S[~mask], marker="x",
            color="grey", label="Excluded")

plt.plot(ell, S_fit, "r-", label=f"Fit: c = {c:.4f}")

plt.xlabel("Bond position")
plt.ylabel("Entanglement entropy")
plt.legend()
plt.show()

# Plot 2: Calabrese-Cardy fit
plt.figure()

plt.scatter(x[mask], S[mask], s=15, label="Used in fit")
plt.scatter(x[~mask], S[~mask], marker="x",
            color="grey", label="Excluded")

plt.plot(x, S_fit, "r-", label=f"Fit: c = {c:.4f}")

plt.xlabel(r"$\ln\left[\frac{2L}{\pi}\sin\left(\frac{\pi\ell}{L}\right)\right]$")
plt.ylabel("Entanglement entropy")
plt.legend()
plt.show()