import numpy as np

from dmrg.mps import product_state
from dmrg.tfim import tfim_mpo
from dmrg.sweep import sweep

L = 5
J = 1.0
h = 0.5
bond_dim = 4  # for full capture of an arbitrary state, need d^(L/2) 

mps = product_state([1.0, 0.0], L)
mpo = tfim_mpo(L, J, h)

final_mps, ground_energy = sweep(
    mps, mpo,
    bond_dim=bond_dim,
    number_of_sweeps=20,
    tol=1e-12
)

print("Bond Dimension:", bond_dim)
print("DMRG Ground Energy:", ground_energy)

# H = mpo.contract()
# exact_energy = np.linalg.eigvalsh(H)[0]
# print("Exact:", exact_energy)
# print("Difference:", abs(ground_energy - exact_energy))