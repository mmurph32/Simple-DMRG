import numpy as np

# Ineffcient
# def energy(mps, mpo):
#     psi = mps.contract()
#     H = mpo.contract()
#     energy = np.vdot(psi, H @ psi) / np.vdot(psi, psi)
#     return energy
import numpy as np
from dmrg.environments import update_left_env

def energy(mps, mpo):

    env = np.ones((1, 1, 1), dtype=complex)

    for i in range(mps.L):
        env = update_left_env(
            env,
            mps.tensors[i],
            mpo.mpo_tensors[i]
        )

    return np.real_if_close(env[0, 0, 0])