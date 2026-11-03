
import numpy as np
from dmrg.mpo import SimpleMPO


def constrained_hard_boson_mpo(L, w, U, V):
    """
    Exact projected hard-boson Hamiltonian.

    H = -w sum_i P_(i-1) X_i P_(i+1)
        + U sum_i n_i
        + V sum_i n_i n_(i+2)

    Open boundaries.
    Local basis: |0>, |1>.

    MPO shape: (D_left, d_in, d_out, D_right).
    """

    if L < 3:
        raise ValueError("L must be >= 3")

    I = np.eye(2)
    X = np.array([[0., 1.], [1., 0.]])
    n = np.diag([0., 1.])
    P = I - n

    # Virtual states:
    # 0: idle
    # 1: started P-X-P term with P
    # 2: completed P-X, waiting for P
    # 3: started n-I-n term with n
    # 4: completed n-I, waiting for n
    # 5: finished
    # 6: boundary X-P term, waiting for P

    D = 7

    first = np.zeros((1, 2, 2, D))
    bulk = np.zeros((D, 2, 2, D))
    last = np.zeros((D, 2, 2, 1))

    # First site
    first[0, :, :, 0] = I
    first[0, :, :, 1] = P
    first[0, :, :, 3] = n
    first[0, :, :, 5] = U * n
    first[0, :, :, 6] = -w * X

    # Bulk
    bulk[0, :, :, 0] = I
    bulk[0, :, :, 1] = P
    bulk[0, :, :, 3] = n
    bulk[0, :, :, 5] = U * n

    # P-X-P
    bulk[1, :, :, 2] = -w * X
    bulk[2, :, :, 5] = P

    # n-I-n
    bulk[3, :, :, 4] = I
    bulk[4, :, :, 5] = V * n

    # Completed terms
    bulk[5, :, :, 5] = I

    # Left-boundary X-P
    bulk[6, :, :, 5] = P

    # Last site
    last[0, :, :, 0] = U * n
    last[1, :, :, 0] = -w * X
    last[2, :, :, 0] = P
    last[4, :, :, 0] = V * n
    last[5, :, :, 0] = I
    last[6, :, :, 0] = P

    return SimpleMPO(
        [first] + [bulk.copy() for _ in range(L - 2)] + [last]
    )
