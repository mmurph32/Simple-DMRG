import numpy as np
from dmrg.effective_hamiltonian import effective_action
from scipy.sparse.linalg import LinearOperator, eigsh
from dmrg.fast_eff_action import effective_action_fast
from dmrg.effective_hamiltonian import effective_action_matrix_free

# def eigensolver(mps, mpo, centre, theta):

#     theta_shape = theta.shape

#     def apply_H(vector):
#         theta = vector.reshape(theta_shape)
#         result = effective_action(mps, mpo, centre, theta)
#         return result.reshape(-1)


#     dim = theta.size

#     H = LinearOperator(
#         shape=(dim, dim),
#         matvec=apply_H
#     )

#     e_value, e_vector = eigsh(H, k=1, which="SA")

#     e_vector = e_vector[:, 0].reshape(theta_shape)

#     return e_value[0], e_vector

# from dmrg.environments import build_environments

# def eigensolver(mps, mpo, centre, theta):

#     theta_shape = theta.shape

#     L, R = build_environments(mps, mpo, centre)

#     def apply_H(vector):
#         theta = vector.reshape(theta_shape)
#         result = effective_action(mpo, centre, theta, L, R)
#         return result.reshape(-1)

#     dim = theta.size

#     H = LinearOperator(
#         shape=(dim, dim),
#         matvec=apply_H
#     )

#     e_value, e_vector = eigsh(H, k=1, which="SA")

#     e_vector = e_vector[:, 0].reshape(theta_shape)

#     return e_value[0], e_vector


def eigensolver(mps, mpo, centre, theta, L_env, R_env):

    theta_shape = theta.shape

    def apply_H(vector):
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

    e_value, e_vector = eigsh(H, k=1, which="SA")

    e_vector = e_vector[:, 0].reshape(theta_shape)

    return e_value[0], e_vector
