
import numpy as np
import matplotlib.pyplot as plt

from dmrg.mps import product_state, SimpleMPS
from dmrg.sweep import sweep
from dmrg.entanglement import entanglement_entropy
from dmrg.mpo import SimpleMPO
from dmrg.eigensolver import eigensolver, right_canonicalize
from dmrg.environments import update_left_env, update_right_env

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

def idmrg_snapshots(
    mpo3,
    phi,
    chi_max,
    L_values,
    svd_min=1e-12,
    verbose=True
):
    """
    Grow a chain once and save MPS snapshots at requested lengths.

    Returns
    -------
    snapshots : dict
        {L: SimpleMPS}
    history : list
        Growth energies and bond dimensions.
    """

    L_values = sorted(set(L_values))

    if not L_values or any(L < 2 or L % 2 for L in L_values):
        raise ValueError("All requested lengths must be even and >= 2")

    max_sites = max(L_values)

    W_first, W_bulk, W_last = mpo3.mpo_tensors

    dtype = np.result_type(np.asarray(phi).dtype, W_bulk.dtype, np.float64)
    phi = np.asarray(phi, dtype=dtype)
    phi /= np.linalg.norm(phi)

    d = len(phi)

    left_env = np.ones((1, 1, 1), dtype=dtype)
    right_env = np.ones((1, 1, 1), dtype=dtype)

    schmidt = np.ones(1)

    first_window = SimpleMPO([W_first, W_last])
    bulk_window = SimpleMPO([W_bulk, W_bulk])

    left_sites = []
    right_sites = []

    snapshots = {}
    history = []

    for step in range(max_sites // 2):

        if step == 0:
            W_l, W_r, window = W_first, W_last, first_window
        else:
            W_l, W_r, window = W_bulk, W_bulk, bulk_window

        chi_old = len(schmidt)

        theta = np.einsum(
            "ab,s,t->astb",
            np.diag(schmidt), phi, phi
        )

        _, optimized = eigensolver(
            None, window, 0, theta,
            left_env, right_env
        )

        U, s, Vh = np.linalg.svd(
            optimized.reshape(chi_old * d, d * chi_old),
            full_matrices=False
        )

        s_norm = np.linalg.norm(s)

        chi_new = min(
            chi_max,
            max(1, np.count_nonzero(s / s_norm >= svd_min))
        )

        schmidt = s[:chi_new]
        schmidt /= np.linalg.norm(schmidt)

        A = U[:, :chi_new].reshape(chi_old, d, chi_new)
        B = Vh[:chi_new, :].reshape(chi_new, d, chi_old)

        left_env = update_left_env(left_env, A, W_l)
        right_env = update_right_env(right_env, B, W_r)

        left_sites.append(A)
        right_sites.insert(0, B)

        L = 2 * (step + 1)

        E = np.einsum(
            "awb,awb,a,b->",
            left_env, right_env, schmidt, schmidt
        ).real.item()

        history.append({
            "L": L,
            "E": E,
            "chi": chi_new
        })

        # Save independent MPS snapshot
        if L in L_values:

            tensors = (
                left_sites[:-1]
                + [left_sites[-1] * schmidt[None, None, :]]
                + right_sites
            )

            snapshots[L] = SimpleMPS(
                [T.copy() for T in tensors]
            )

        if verbose and (L % 10 == 0 or L in L_values):
            print(
                f"Growth L={L}/{max_sites}, "
                f"chi={chi_new}, E={E:.10f}",
                flush=True
            )

    return snapshots, history


def central_charge_convergence_grow_and_save(
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

    mpo3 = mpo_builder(3)

    for chi in chi_values:

        snapshots, history = idmrg_snapshots(
            mpo3=mpo3,
            phi=[1.0, 0.0],
            chi_max=chi,
            L_values=L_values,
            verbose=True
        )

        for L in L_values:

            print(f"Polishing L={L}, chi={chi}")

            mps = right_canonicalize(snapshots[L])
            mpo = mpo_builder(L)

            mps, E = sweep(
                mps,
                mpo,
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