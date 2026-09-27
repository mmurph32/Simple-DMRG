import numpy as np

from dmrg.mps import product_state
from dmrg.tfim import tfim_mpo
from dmrg.environments import build_environments


L = 4
J = 1.0
h = 0.5

phi = np.array([1.0, 0.0])

mps = product_state(phi, L)
mpo = tfim_mpo(L, J, h)

centre = 2

left_env = build_environments(mps, mpo, centre)

print("Left environment shape:", left_env.shape)
print("Left environment:")
print(left_env)