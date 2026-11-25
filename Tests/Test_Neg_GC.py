
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from AGS_Anyon_Model.anyon_idmrg import anyon_grow_and_sweep
from dmrg.entanglement import entanglement_entropy

# Pure antiferromagnetic golden chain
Jeff = -1.0
chi = 32

# L counts fusion labels, N = L - 1 physical anyons
lengths = [144, 168, 192, 216, 240]

N_values = []
S_values = []

for L in lengths:

    print(f"\nRunning L={L}, N={L-1}, chi={chi}")

    mps, E, E_before, history = anyon_grow_and_sweep(
        jeff=Jeff,
        chi_max=chi,
        max_sites=L,
        penalty=50,
        number_of_sweeps=2,
        tol=1e-8,
        verbose=True,
        x_final=0
    )

    S = np.array(entanglement_entropy(mps))

    # Physical number of anyons
    N = L - 1

    # Entanglement cut closest to N/2 physical anyons
    # Fusion label x_k is stored at MPS site k
    k = N // 2
    S_mid = S[k]

    N_values.append(N)
    S_values.append(S_mid)

    print(f"E = {E:.10f}, S_half = {S_mid:.8f}")

    #np.save(f"golden_entropy_L{L}.npy", S)

# Fit S(N/2) = (c/6) ln(N) + constant
slope, intercept = np.polyfit(
    np.log(N_values), S_values, 1
)

c = 6 * slope

print(f"\nCentral charge: c = {c:.6f}")
print("Expected: c = 0.700000")

# Plot
x = np.log(N_values)

plt.figure(figsize=(7, 5))
plt.scatter(x, S_values, label="DMRG")

plt.plot(
    x,
    slope*x + intercept,
    "r-",
    label=f"Fit: c = {c:.4f}"
)

plt.xlabel(r"$\ln N$")
plt.ylabel(r"$S(N/2)$")
plt.title("Antiferromagnetic Fibonacci golden chain")
plt.legend()
plt.show()
