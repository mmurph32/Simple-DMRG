import numpy as np

# def truncation(theta, bond_dim, direction):

#     A_L, d1, d2, A_R = theta.shape

#     theta_matrix = theta.reshape(A_L*d1, d2*A_R)
#     U, S, Vh = np.linalg.svd(theta_matrix, full_matrices=False)

#     chi = min(bond_dim, len(S))

#     U = U[:, :chi]
#     S = S[:chi]
#     Vh = Vh[:chi, :]

#     if direction == "right":

#         theta1 = U
#         theta2 = S[:, None] * Vh

#     elif direction == "left":

#         theta1 = U * S[None, :]
#         theta2 = Vh

#     else:
#         raise ValueError("direction must be 'right' or 'left'")


#     theta1 = theta1.reshape(A_L, d1, chi)
#     theta2 = theta2.reshape(chi, d2, A_R)

#     return theta1, theta2



def truncation(theta, bond_dim, direction, svd_min=1e-12):

    chi_L, d1, d2, chi_R = theta.shape

    theta_matrix = theta.reshape(chi_L * d1, d2 * chi_R)

    U, S, Vh = np.linalg.svd(
        theta_matrix, full_matrices=False
    )

    # Normalise singular values for deciding truncation
    S_norm = S / np.linalg.norm(S)

    # Keep only significant Schmidt values, up to chi_max
    chi = min(
        bond_dim,
        max(1, np.count_nonzero(S_norm >= svd_min))
    )

    # Discarded weight
    discarded_weight = np.sum(S_norm[chi:]**2)

    U = U[:, :chi]
    S = S[:chi]
    Vh = Vh[:chi, :]

    # Renormalise after truncation
    S = S / np.linalg.norm(S)

    if direction == "right":
        A = U
        B = S[:, None] * Vh

    elif direction == "left":
        A = U * S[None, :]
        B = Vh

    else:
        raise ValueError("Invalid direction")

    A = A.reshape(chi_L, d1, chi)
    B = B.reshape(chi, d2, chi_R)

    return A, B