
import numpy as np
from scipy.linalg import eigh_tridiagonal


def lanczos_ground_state(matvec, v0, tol=1e-8, maxiter=60):

    q = np.asarray(v0).reshape(-1).copy()
    q /= np.linalg.norm(q)

    basis = []
    alphas = []
    betas = []

    q_prev = np.zeros_like(q)
    beta_prev = 0.0

    for iteration in range(min(maxiter, q.size)):

        basis.append(q)

        # Apply effective Hamiltonian
        w = matvec(q)

        alpha = np.vdot(q, w).real
        w = w - alpha * q - beta_prev * q_prev

        # Reorthogonalisation for numerical stability
        for b in basis:
            w -= np.vdot(b, w) * b

        beta = np.linalg.norm(w)
        alphas.append(alpha)

        # Diagonalise the small Lanczos matrix
        values, vectors = eigh_tridiagonal(alphas, betas)

        E = values[0]
        y = vectors[:, 0]

        # Ritz residual estimates eigenvector convergence
        residual = beta * abs(y[-1])

        if residual < tol:

            psi = sum(
                y[j] * basis[j]
                for j in range(len(basis))
            )

            psi /= np.linalg.norm(psi)

            return E, psi, iteration + 1, residual

        betas.append(beta)
        q_prev, q = q, w / beta
        beta_prev = beta

    raise RuntimeError(
        f"Lanczos did not converge. Residual = {residual:.2e}"
    )
