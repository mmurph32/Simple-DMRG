import cProfile
import pstats
import time

from dmrg.mps import product_state
from dmrg.tfim import tfim_mpo
from dmrg.sweep import sweep
from dmrg.idmrg import idmrg
from dmrg.grow_and_sweep import grow_and_sweep

from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS


# Parameters
L = 100
J = 1.0
h = 1.0
chi = 64
max_sweeps = 20
tol = 1e-10

mpo = tfim_mpo(L, J, h)
mpo3 = tfim_mpo(3, J, h)

results = {}


# 1. Your finite DMRG
mps = product_state([1.0, 0.0], L)
profiler = cProfile.Profile()
profiler.enable()

mps, E = sweep(
    mps, mpo,
    bond_dim=chi,
    number_of_sweeps=max_sweeps,
    tol=tol
)

profiler.disable()

print("\n--- Eigensolver profiling ---")
pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)


# 4. TeNPy finite DMRG
print("\n--- TeNPy Finite DMRG ---", flush=True)

model = TFIChain({
    "L": L,
    "J": J,
    "g": h,
    "bc_MPS": "finite",
    "conserve": None
})

psi = MPS.from_product_state(
    model.lat.mps_sites(),
    ["up"] * L,
    bc="finite"
)

options = {
    "active_sites": 2,
    "mixer": None,
    "N_sweeps_check": 1,
    "min_sweeps": 1,
    "max_sweeps": max_sweeps,
    "max_E_err": tol,
    "max_S_err": float("inf"),
    "trunc_params": {
        "chi_max": chi,
        "svd_min": 1e-12,
        "trunc_cut": 1e-14
    }
}

start = time.perf_counter()
info = dmrg.run(psi, model, options)

results["TeNPy"] = (
    info["E"],
    time.perf_counter() - start,
    max(psi.chi)
)


# Results
print("\n" + "=" * 75)
print(f"L = {L}, chi_max = {chi}, h = {h}")
print("=" * 75)

print(f"{'Method':<22} {'Energy':>19} {'Time (s)':>12} {'Max chi':>10}")
print("-" * 75)

for name, (E, elapsed, chi_actual) in results.items():
    print(
        f"{name:<22} {E:>19.12f} "
        f"{elapsed:>12.3f} {chi_actual:>10}"
    )

print("-" * 75)

E_ref = results["TeNPy"][0]

for name, (E, _, _) in results.items():
    print(
        f"{name:<22} energy error vs TeNPy: "
        f"{abs(E-E_ref):.3e}"
    )

print("\nTeNPy sweeps:",
      info["sweep_statistics"]["sweep"][-1])
