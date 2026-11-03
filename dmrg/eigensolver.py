import numpy as np
from dmrg.effective_hamiltonian import effective_action
from scipy.sparse.linalg import LinearOperator, eigsh
from dmrg.fast_eff_action import effective_action_fast
from dmrg.effective_hamiltonian import effective_action_matrix_free
from dmrg.lanczos import lanczos_ground_state


def right_canonicalize(mps):

    for i in range(mps.L - 1, 0, -1):

        A = mps.tensors[i]
        chi_L, d, chi_R = A.shape

        Q, R = np.linalg.qr(
            A.reshape(chi_L, d * chi_R).T,
            mode="reduced"
        )

        chi_new = Q.shape[1]

        mps.tensors[i] = Q.T.reshape(chi_new, d, chi_R)

        mps.tensors[i - 1] = np.tensordot(
            mps.tensors[i - 1],
            R.T,
            axes=(2, 0)
        )

    # The first tensor now contains the overall norm.
    mps.tensors[0] /= np.linalg.norm(mps.tensors[0])

    return mps

def eigensolver(mps, mpo, centre, theta, L_env, R_env, tol=1e-12, ncv=12, debug=False):

    theta_shape = theta.shape

    n_matvec = 0

    def apply_H(vector):
        nonlocal n_matvec
        n_matvec += 1

        theta_trial = vector.reshape(theta_shape)

        result = effective_action_matrix_free(
            mpo, centre, theta_trial, L_env, R_env
        )

        return result.reshape(-1)

    dim = theta.size
    
    H = LinearOperator(
        shape=(dim, dim),
        matvec=apply_H
    )

    e_value, e_vector = eigsh(
    H,
    k=1,
    which="SA",
    v0=theta.reshape(-1),
    tol=tol,
    ncv=min(ncv, dim)
    )

    e_vector = e_vector[:, 0].reshape(theta_shape)

    return e_value[0], e_vector
