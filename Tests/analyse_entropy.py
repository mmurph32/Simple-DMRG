import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from dmrg.central_charge import fit_central_charge

# Load entropy
S = np.load("hard_boson_entropy.npy")
L = len(S) + 1

# Calabrese-Cardy fit
c, slope, intercept, x = fit_central_charge(S, trim=0)

print(f"Central charge: c = {c:.6f}")

ell = np.arange(1, L)
S_fit = slope * x + intercept

# Plot 1: Entropy vs bond position
plt.figure()
plt.scatter(ell, S, label="DMRG")
plt.plot(ell, S_fit, "r-", label=f"Fit: c = {c:.4f}")
plt.xlabel("Bond position")
plt.ylabel("Entanglement entropy")
plt.legend()
plt.show()


# Plot 2: Calabrese-Cardy linear fit
plt.figure()
plt.scatter(x, S, label="DMRG")
plt.plot(x, S_fit, "r-", label=f"Fit: c = {c:.4f}")
plt.xlabel(r"$\ln\left[\frac{2L}{\pi}\sin\left(\frac{\pi\ell}{L}\right)\right]$")
plt.ylabel("Entanglement entropy")
plt.legend()
plt.show()
