
import time
import numpy as np
import matplotlib.pyplot as plt

# Your DMRG
from dmrg.mps import product_state
from dmrg.tfim import tfim_mpo
from dmrg.sweep import sweep
from dmrg.entanglement import entanglement_entropy

# TeNPy
from tenpy.algorithms import dmrg
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS


# %% Parameters

L = 50
J = 1.0
h = 1.0
chi = 16
n_sweeps = 5


# %% Your DMRG

mps = product_state([1.0, 0.0], L)
mpo = tfim_mpo(L, J, h)

start = time.perf_counter()

final_mps, E_mine = sweep(
    mps, mpo,
    bond_dim=chi,
    number_of_sweeps=n_sweeps,
    tol=0
)

time_mine = time.perf_counter() - start

S_mine = entanglement_entropy(final_mps)


# %% TeNPy DMRG

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

dmrg_params = {
    "mixer": None,
    "min_sweeps": n_sweeps,
    "max_sweeps": n_sweeps,
    "max_E_err": 1e-10,
    "trunc_params": {
        "chi_max": chi,
        "svd_min": 1e-12
    }
}

start = time.perf_counter()

info = dmrg.run(psi, model, dmrg_params)

time_tenpy = time.perf_counter() - start

E_tenpy = info["E"]
S_tenpy = psi.entanglement_entropy()


# %% Results

print("\n" + "=" * 45)
print(f"L = {L}, chi = {chi}, sweeps = {n_sweeps}")
print("=" * 45)

print(f"My DMRG energy:   {E_mine:.12f}")
print(f"TeNPy energy:     {E_tenpy:.12f}")
print(f"Energy difference: {abs(E_mine-E_tenpy):.3e}")

print(f"\nMy runtime:       {time_mine:.3f} s")
print(f"TeNPy runtime:    {time_tenpy:.3f} s")
print(f"Runtime ratio:    {time_mine/time_tenpy:.2f}x")

print(f"\nMy max chi:       {max(A.shape[2] for A in final_mps.tensors)}")
print(f"TeNPy max chi:    {max(psi.chi)}")


# %% Entanglement comparison

ell = np.arange(1, L)

plt.figure()
plt.plot(ell, S_mine, "o-", label="My DMRG")
plt.plot(ell, S_tenpy, "x--", label="TeNPy")

plt.xlabel("Bond position")
plt.ylabel("Entanglement entropy")
plt.title(f"TFIM: L={L}, chi={chi}")
plt.legend()
plt.tight_layout()
plt.show()

print("\nMaximum entropy difference:",
      np.max(np.abs(S_mine - S_tenpy)))

