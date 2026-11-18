import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

from dmrg.mps import product_state
from dmrg.tfim import tfim_mpo
from dmrg.sweep import sweep

from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS


# Parameters
L = 200
J = 1.0
h = 1.0
chi = 128
tol = 1e-10
max_sweeps = 20


# 1. Your DMRG with Lanczos
print("\n--- My Lanczos DMRG ---", flush=True)

mpo = tfim_mpo(L, J, h)
mps = product_state([1.0, 0.0], L)

start = time.perf_counter()

mps, E_mine = sweep(
    mps, mpo,
    bond_dim=chi,
    number_of_sweeps=max_sweeps,
    tol=tol
)

time_mine = time.perf_counter() - start
chi_mine = max(A.shape[2] for A in mps.tensors)


# 2. TeNPy finite DMRG
print("\n--- TeNPy DMRG ---", flush=True)

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

time_tenpy = time.perf_counter() - start
E_tenpy = info["E"]


# 3. Results
print("\n" + "=" * 50)
print(f"L={L}, chi_max={chi}")
print("=" * 50)

print(f"My Lanczos energy:  {E_mine:.12f}")
print(f"TeNPy energy:       {E_tenpy:.12f}")
print(f"Energy difference:  {abs(E_mine-E_tenpy):.3e}")

print(f"\nMy runtime:         {time_mine:.3f} s")
print(f"TeNPy runtime:      {time_tenpy:.3f} s")
print(f"Runtime ratio:      {time_mine/time_tenpy:.2f}x")

print(f"\nMy maximum chi:     {chi_mine}")
print(f"TeNPy maximum chi:  {max(psi.chi)}")

stats = info["bond_statistics"]
print(f"\nTeNPy sweeps:       {info['sweep_statistics']['sweep'][-1]}")
print(f"TeNPy avg Lanczos:  {sum(stats['N_lanczos'])/len(stats['N_lanczos']):.2f}")
