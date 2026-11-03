
import numpy as np
from dmrg.mpo import SimpleMPO


def hard_boson_mpo(L, w, U, V, penalty):

    """
    Soft-penalty approximation to the Fendley hard-boson model.

    H = -w sum(X_i)
        + U sum(n_i)
        + V sum(n_i n_{i+2})
        + penalty sum(n_i n_{i+1})

    Local basis: |0>, |1>

    Open boundary conditions.
    """

    if L < 3:
        raise ValueError("L must be at least 3")

    # Local operators
    I = np.eye(2)

    X = np.array([
        [0, 1],
        [1, 0]
    ])

    n = np.array([
        [0, 0],
        [0, 1]
    ])

    # On-site Hamiltonian
    onsite = -w * X + U * n

    # MPO bond dimension
    D = 5

    # First, bulk and last MPO tensors
    first = np.zeros((1, 2, 2, D))
    bulk = np.zeros((D, 2, 2, D))
    last = np.zeros((D, 2, 2, 1))

    # ------------------------------------------------
    # FIRST SITE
    # ------------------------------------------------

    first[0, :, :, 0] = I
    first[0, :, :, 1] = n
    first[0, :, :, 2] = n
    first[0, :, :, 4] = onsite

    # ------------------------------------------------
    # BULK SITES
    # ------------------------------------------------

    # Continue idle state
    bulk[0, :, :, 0] = I

    # Start nearest-neighbour interaction
    bulk[0, :, :, 1] = n

    # Complete nearest-neighbour interaction
    bulk[1, :, :, 4] = penalty * n

    # Start next-nearest-neighbour interaction
    bulk[0, :, :, 2] = n

    # Insert identity between next-nearest neighbours
    bulk[2, :, :, 3] = I

    # Complete next-nearest-neighbour interaction
    bulk[3, :, :, 4] = V * n

    # On-site terms
    bulk[0, :, :, 4] = onsite

    # Continue completed terms
    bulk[4, :, :, 4] = I

    # ------------------------------------------------
    # LAST SITE
    # ------------------------------------------------

    last[0, :, :, 0] = onsite
    last[1, :, :, 0] = penalty * n
    last[3, :, :, 0] = V * n
    last[4, :, :, 0] = I

    # Assemble finite open-chain MPO
    tensors = (
        [first]
        + [bulk] * (L - 2)
        + [last]
    )

    return SimpleMPO(tensors)
