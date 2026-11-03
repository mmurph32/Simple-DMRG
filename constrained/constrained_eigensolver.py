
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

from dmrg.effective_hamiltonian import effective_action_matrix_free


def constrained_eigensolver(
    mpo, centre, theta, L_env, R_env, qL, qR,
    tol=1e-12
):
    chiL, d1, d2, chiR = theta.shape

    # Identify allowed components
    mask = np.zeros(theta.shape, dtype=bool)

    for a, left in enumerate(qL):
        for n1 in range(d1):
            for n2 in range(d2):
                for b, right in enumerate(qR):

                    allowed = (
                        left * n1 == 0
                        and n1 * n2 == 0
                        and (right == n2 or right == -1)
                    )

                    mask[a, n1, n2, b] = allowed

    indices = np.flatnonzero(mask.ravel())
    dim = len(indices)

    if dim == 0:
        raise ValueError("No allowed two-site configurations")

    full = np.zeros(theta.size, dtype=theta.dtype)

    def apply_H(v):
        full.fill(0)
        full[indices] = v

        result = effective_action_matrix_free(
            mpo, centre,
            full.reshape(theta.shape),
            L_env, R_env
        )

        return result.ravel()[indices]

    v0 = theta.ravel()[indices].copy()

    if np.linalg.norm(v0) == 0:
        v0 = np.ones(dim, dtype=theta.dtype)

    v0 /= np.linalg.norm(v0)

    # Small problems: exact local diagonalisation
    if dim <= 3:
        H = np.column_stack([
            apply_H(np.eye(dim, dtype=theta.dtype)[:, j])
            for j in range(dim)
        ])
        H = (H + H.conj().T) / 2
        values, vectors = eigh(H)
        E, psi = values[0], vectors[:, 0]

    else:
        H = LinearOperator(
            (dim, dim),
            matvec=apply_H,
            dtype=theta.dtype
        )

        values, vectors = eigsh(
            H, k=1, which="SA",
            v0=v0, tol=tol,
            ncv=min(12, dim)
        )

        E, psi = values[0], vectors[:, 0]

    theta_new = np.zeros(theta.size, dtype=theta.dtype)
    theta_new[indices] = psi

    return E, theta_new.reshape(theta.shape)
