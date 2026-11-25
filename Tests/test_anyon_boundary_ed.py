
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from AGS_Anyon_Model.anyon_idmrg import anyon_grow_and_sweep
from dmrg.entanglement import entanglement_entropy

# Parameters
Jeff = -1.0              # +1 -> c=0.8, -1 -> c=0.7
chi = 32
penalty = 50

lengths = np.array([80, 104, 128, 152, 176, 200, 224, 248])

N_values = []
S_values = []

# Pure golden chain: ring term must be disabled
for L in lengths:

    print(f"\nRunning L={L}, chi={chi}")

    mps, E, _, _ = anyon_grow_and_sweep(
        jeff=Jeff,
        chi_max=chi,
        max_sites=L,
        penalty=penalty,
        number_of_sweeps=2,
        tol=1e-8,
        verbose=True,
        x_final=0
    )

    S = np.asarray(entanglement_entropy(mps))

    N = L - 1
    S_mid = S[L//2 - 1]

    N_values.append(N)
    S_values.append(S_mid)

    print(f"L={L}, E={E:.8f}, S_mid={S_mid:.8f}")

# Fit S(N/2) = (c/6) ln(N) + constant
N_values = np.array(N_values)
S_values = np.array(S_values)

# Fit using progressively larger systems
print("\nCentral-charge estimates:")

for start in range(len(N_values) - 3):

    x = np.log(N_values[start:])
    y = S_values[start:]

    slope, intercept = np.polyfit(x, y, 1)
    c = 6 * slope

    print(f"N >= {N_values[start]}: c = {c:.6f}")

# Fit using the five largest systems
x = np.log(N_values[-5:])
y = S_values[-5:]

slope, intercept = np.polyfit(x, y, 1)
c = 6 * slope

print(f"\nFinal central charge (largest 5): c = {c:.6f}")

# Plot
plt.figure(figsize=(7, 5))

plt.scatter(np.log(N_values), S_values,
            color="black", label="DMRG")

x_fit = np.linspace(x.min(), x.max(), 100)

plt.plot(
    x_fit,
    slope*x_fit + intercept,
    "r-",
    label=f"Fit: c = {c:.4f}"
)

plt.xlabel(r"$\ln N$")
plt.ylabel(r"$S(N/2)$")
plt.title(f"Golden chain: Jeff={Jeff}, chi={chi}")
plt.legend()
plt.tight_layout()
plt.show()

# Save results
np.savez(
    f"golden_scaling_Jeff{Jeff}_chi{chi}.npz",
    N=N_values,
    S=S_values
)
