
import numpy as np
import matplotlib.pyplot as plt

from dmrg.mps import product_state
from dmrg.sweep import sweep
from dmrg.entanglement import entanglement_entropy

def extract_central_charge(entropies):
    L = len(entropies) + 1

    L_range = np.arange(1, L)

    # Calabrese-Cardy conformal distance
    x = np.log(
        (2 * L / np.pi) * np.sin(np.pi * L_range / L)
    )

    # Fit S = slope*x + intercept
    slope, intercept = np.polyfit(x, entropies, 1)

    c = 6 * slope

    return c, slope, intercept


import numpy as np


def fit_central_charge(S, trim=5):
    """
    Calabrese-Cardy fit for an open chain.
    No smoothing or assumed oscillation period.
    """

    S = np.asarray(S)
    L = len(S) + 1

    ell = np.arange(1, L)

    x = np.log(
        (2 * L / np.pi) *
        np.sin(np.pi * ell / L)
    )

    mask = (ell > trim) & (ell < L - trim)

    slope, intercept = np.polyfit(
        x[mask], S[mask], 1
    )

    c = 6 * slope

    return c, slope, intercept, x



def central_charge_convergence(
    mpo_builder,
    L_values,
    chi_values,
    trim_max=0.4,
    trim_steps=21,
    sweeps=20,
    tol=1e-10,
    expected_c=None
):
    """
    Study central-charge convergence with L, chi and boundary trim.

    mpo_builder(L) must return the Hamiltonian MPO.
    """

    trims = np.linspace(0, trim_max, trim_steps)
    results = {}

    plt.figure(figsize=(9, 6))

    for L in L_values:
        for chi in chi_values:

            print(f"\nRunning L={L}, chi={chi}", flush=True)

            mpo = mpo_builder(L)
            mps = product_state([1.0, 0.0], L)

            mps, E = sweep(
                mps, mpo,
                bond_dim=chi,
                number_of_sweeps=sweeps,
                tol=tol,
                verbose=False
            )

            S = entanglement_entropy(mps)

            c_values = []

            for fraction in trims:
                trim = int(fraction * L)

                if L - 2 * trim < 8:
                    c_values.append(np.nan)
                    continue

                c, _, _, _ = fit_central_charge(
                    S, trim=trim
                )

                c_values.append(c)

            results[(L, chi)] = {
                "energy": E,
                "entropy": S,
                "trims": trims,
                "c": np.array(c_values)
            }

    colors = plt.cm.tab10(np.linspace(0, 1, len(L_values)))

    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]

    for i, L in enumerate(L_values):
        for j, chi in enumerate(chi_values):

            data = results[(L, chi)]

            plt.plot(
                data["trims"] * 100,
                data["c"],
                color=colors[i],
                linestyle=linestyles[j % len(linestyles)],
                marker=markers[j % len(markers)],
                markevery=(j % 5, 5),
                markersize=5,
                linewidth=1.5,
                label=f"L={L}, chi={chi}"
            )

    if expected_c is not None:
        plt.axhline(
            expected_c,
            color="black",
            linestyle="--",
            label=f"Expected c={expected_c}"
        )

    plt.xlabel("Boundary trim (% per side)")
    plt.ylabel("Fitted central charge")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results