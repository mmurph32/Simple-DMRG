
import numpy as np

from dmrg.energy import energy
from dmrg.environments import update_left_env, update_right_env
from dmrg.constrained_eigensolver import constrained_eigensolver
from dmrg.constrained_truncation import constrained_truncation


def constrained_sweep(
    mps, mpo, bond_dim=32,
    number_of_sweeps=10, tol=1e-10,
    verbose=True
):

    N = mps.L
    E_prev = energy(mps, mpo)

    right_envs = [None] * (N + 1)
    right_envs[N] = np.ones((1, 1, 1))

    for i in range(N - 1, -1, -1):
        right_envs[i] = update_right_env(
            right_envs[i+1],
            mps.tensors[i],
            mpo.mpo_tensors[i]
        )

    for sweep_no in range(number_of_sweeps):

        left_envs = [None] * (N + 1)
        left_envs[0] = np.ones((1, 1, 1))

        # Left -> right
        for i in range(N - 1):

            theta = np.tensordot(
                mps.tensors[i],
                mps.tensors[i+1],
                axes=(-1, 0)
            )

            _, theta_new = constrained_eigensolver(
                mpo, i, theta,
                left_envs[i], right_envs[i+2],
                mps.sectors[i], mps.sectors[i+2]
            )

            A, B, q_mid = constrained_truncation(
                theta_new,
                mps.sectors[i], mps.sectors[i+2],
                bond_dim, "right"
            )

            mps.tensors[i] = A
            mps.tensors[i+1] = B
            mps.sectors[i+1] = q_mid

            left_envs[i+1] = update_left_env(
                left_envs[i],
                A, mpo.mpo_tensors[i]
            )

        # Right -> left
        right_envs[N] = np.ones((1, 1, 1))

        for i in range(N - 1, 0, -1):

            theta = np.tensordot(
                mps.tensors[i-1],
                mps.tensors[i],
                axes=(-1, 0)
            )

            _, theta_new = constrained_eigensolver(
                mpo, i-1, theta,
                left_envs[i-1], right_envs[i+1],
                mps.sectors[i-1], mps.sectors[i+1]
            )

            A, B, q_mid = constrained_truncation(
                theta_new,
                mps.sectors[i-1], mps.sectors[i+1],
                bond_dim, "left"
            )

            mps.tensors[i-1] = A
            mps.tensors[i] = B
            mps.sectors[i] = q_mid

            right_envs[i] = update_right_env(
                right_envs[i+1],
                B, mpo.mpo_tensors[i]
            )

        E_new = energy(mps, mpo)

        if verbose:
            print(
                f"Sweep {sweep_no+1}/{number_of_sweeps} "
                f"E={E_new:.12f}",
                flush=True
            )

        if abs(E_new - E_prev) < tol:
            if verbose:
                print("Converged.")
            break

        E_prev = E_new

    mps.check_constraint()

    return mps, E_new
