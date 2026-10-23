import numpy as np

# Ineffcient
# def energy(mps, mpo):
#     psi = mps.contract()
#     H = mpo.contract()
#     energy = np.vdot(psi, H @ psi) / np.vdot(psi, psi)
#     return energy

def energy(mps, mpo):

    # Left boundary environment
    env = np.ones((1, 1, 1), dtype=complex)

    # Contract from left to right
    for i in range(mps.L):

        A = mps.tensors[i]
        W = mpo.mpo_tensors[i]

        env = np.einsum(
            'abc,asd,bste,ctf->def',
            env, A, W, A.conj(),
            optimize=True
        )

    numerator = env[0, 0, 0]

    # Calculate norm without constructing the state
    norm_env = np.ones((1, 1), dtype=complex)

    for A in mps.tensors:
        norm_env = np.einsum(
            'ab,asc,bsd->cd',
            norm_env, A, A.conj(),
            optimize=True
        )

    denominator = norm_env[0, 0]

    return np.real_if_close(numerator / denominator)
