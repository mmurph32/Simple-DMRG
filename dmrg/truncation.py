import numpy as np

def truncation(theta, bond_dim, direction):

    A_L, d1, d2, A_R = theta.shape

    theta_matrix = theta.reshape(A_L*d1, d2*A_R)
    U, S, Vh = np.linalg.svd(theta_matrix, full_matrices=False)

    chi = min(bond_dim, len(S))

    U = U[:, :chi]
    S = S[:chi]
    Vh = Vh[:chi, :]

    if direction == "right":

        theta1 = U
        theta2 = S[:, None] * Vh

    elif direction == "left":

        theta1 = U * S[None, :]
        theta2 = Vh

    else:
        raise ValueError("direction must be 'right' or 'left'")


    theta1 = theta1.reshape(A_L, d1, chi)
    theta2 = theta2.reshape(chi, d2, A_R)

    return theta1, theta2
    