import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt

# Load results
data = np.load(
    "central_charge_results.npz",
    allow_pickle=True
)

#data = np.load("central_charge_results.npz", allow_pickle=True)
data = np.load("hard_boson_c_results.npz", allow_pickle=True)

print("Saved keys:", data.files)
results = data["arr_0"].item()
# Parameters
chi = max(key[1] for key in results)
trim_fraction = 0.2

L_values = sorted(set(key[0] for key in results))

c_values = []

for L in L_values:

    result = results[(L, chi)]

    trims = result["trims"]
    c = result["c"]

    # Closest available trim percentage
    idx = np.argmin(abs(trims - trim_fraction))

    c_values.append(c[idx])

L_values = np.array(L_values)
c_values = np.array(c_values)

# Extrapolate c(L) = c_inf + a/L
x = 1 / L_values

slope, c_inf = np.polyfit(x, c_values, 1)

print(f"Largest chi: {chi}")
print(f"Trim: {trim_fraction*100:.0f}% per side")
print(f"Extrapolated central charge: c = {c_inf:.6f}")
print(f"Expected Ising: c = 0.500000")

# Plot
plt.figure()

plt.scatter(x, c_values, label="DMRG")

x_fit = np.linspace(0, max(x), 200)

plt.plot(
    x_fit,
    slope*x_fit + c_inf,
    "r-",
    label=f"Extrapolated c = {c_inf:.5f}"
)

plt.axhline(
    0.5,
    color="black",
    linestyle="--",
    label="Exact Ising c = 0.5"
)

plt.xlabel(r"$1/L$")
plt.ylabel("Central charge")
plt.legend()
plt.show()
