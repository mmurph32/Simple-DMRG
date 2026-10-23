import numpy as np
from dmrg.eigensolver import eigensolver
from dmrg.truncation import truncation
from dmrg.energy import energy
from dmrg.environments import update_left_env, update_right_env
from tqdm import tqdm

# def sweep(mps, mpo, bond_dim, number_of_sweeps, tol):

#     E_prev = energy(mps, mpo)

#     for n in range(number_of_sweeps):

#         # Sweep left -> right
#         for i in range(0, mps.L - 1):

#             theta = np.tensordot(
#                 mps.tensors[i],
#                 mps.tensors[i + 1],
#                 axes=(-1, 0)
#             )

#             e_val, theta_new = eigensolver(mps, mpo, i, theta)

#             A, B = truncation(theta_new, bond_dim, "right")

#             mps.tensors[i] = A
#             mps.tensors[i + 1] = B


#         # Sweep right -> left
#         for i in range(1, mps.L)[::-1]:

#             theta = np.tensordot(
#                 mps.tensors[i - 1],
#                 mps.tensors[i],
#                 axes=(-1, 0)
#             )

#             e_val, theta_new = eigensolver(mps, mpo, i - 1, theta)

#             A, B = truncation(theta_new, bond_dim, "left")

#             mps.tensors[i - 1] = A
#             mps.tensors[i] = B

#         E_new = energy(mps, mpo)
#         print("Sweep", n+1, "finsihed.")

#         if abs(E_new - E_prev) < tol:
#             print("Tolerance reached after", n+1, "sweeps.")
#             break
#         else:
#             E_prev = E_new


#     return mps, E_new


def sweep(mps, mpo, bond_dim, number_of_sweeps, tol):

    E_prev = energy(mps, mpo)

    N = mps.L

    right_envs = [None] * (N + 1)
    right_envs[N] = np.ones((1, 1, 1))

    for i in range(N - 1, -1, -1):
        right_envs[i] = update_right_env(
            right_envs[i + 1],
            mps.tensors[i],
            mpo.mpo_tensors[i]
        )

    for n in (range(number_of_sweeps)):
        
        left_envs = [None] * (N + 1)
        left_envs[0] = np.ones((1, 1, 1))

        # Sweep left -> right
        for i in range(N - 1):

            theta = np.tensordot(
                mps.tensors[i],
                mps.tensors[i + 1],
                axes=(-1, 0)
            )

            e_val, theta_new = eigensolver(
                mps, mpo, i, theta,
                left_envs[i], right_envs[i + 2]
            )

            A, B = truncation(theta_new, bond_dim, "right")

            mps.tensors[i] = A
            mps.tensors[i + 1] = B

            # Update and save the left environment
            left_envs[i + 1] = update_left_env(
                left_envs[i],
                mps.tensors[i],
                mpo.mpo_tensors[i]
            )

        right_envs[N] = np.ones((1, 1, 1))
        # Sweep right -> left
        for i in range(N - 1, 0, -1):

            theta = np.tensordot(
                mps.tensors[i - 1],
                mps.tensors[i],
                axes=(-1, 0)
            )

            e_val, theta_new = eigensolver(
                mps, mpo, i - 1, theta,
                left_envs[i - 1], right_envs[i + 1]
            )

            A, B = truncation(theta_new, bond_dim, "left")

            mps.tensors[i - 1] = A
            mps.tensors[i] = B

            # Update and save the right environment
            right_envs[i] = update_right_env(
                right_envs[i + 1],
                mps.tensors[i],
                mpo.mpo_tensors[i]
            )

        E_new = energy(mps, mpo)
        print("Sweep", n+1, "finsihed.")

        if abs(E_new - E_prev) < tol:
            print("Tolerance reached after", n+1, "sweeps.")
            break
        else:
            E_prev = E_new


    return mps, E_new