
import numpy as np
from scipy.sparse import csr_matrix


def allowed_states(L):
    """All binary configurations with no adjacent occupied sites."""
    return [
        state for state in range(1 << L)
        if (state & (state >> 1)) == 0
    ]


def constrained_hamiltonian(L, w, U, V):
    """
    Exact Fendley hard-boson Hamiltonian in the
    nearest-neighbour-excluded Hilbert space.

    H = -w sum(P_{i-1} X_i P_{i+1})
        + U sum(n_i)
        + V sum(n_i n_{i+2})
    """
    states = allowed_states(L)
    index = {state: j for j, state in enumerate(states)}

    rows = []
    cols = []
    values = []

    for j, state in enumerate(states):

        occupations = [
            (state >> i) & 1 for i in range(L)
        ]

        # Diagonal terms
        E = U * sum(occupations)

        E += V * sum(
            occupations[i] * occupations[i + 2]
            for i in range(L - 2)
        )

        rows.append(j)
        cols.append(j)
        values.append(E)

        # Constrained creation/annihilation
        for i in range(L):

            new_state = state ^ (1 << i)

            if new_state in index:
                rows.append(index[new_state])
                cols.append(j)
                values.append(-w)

    H = csr_matrix(
        (values, (rows, cols)),
        shape=(len(states), len(states)),
        dtype=float
    )

    return H, states
