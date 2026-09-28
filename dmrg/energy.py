import numpy as np

def energy(mps, mpo):
    psi = mps.contract()
    H = mpo.contract()

    energy = np.vdot(psi, H @ psi) / np.vdot(psi, psi)

    return energy