
import numpy as np
from dmrg.mps import SimpleMPS


class ConstrainedMPS(SimpleMPS):

    def __init__(self, tensors, sectors):
        super().__init__(tensors)
        self.sectors = [np.asarray(q, dtype=int) for q in sectors]
        self.check_constraint()

    def check_constraint(self, atol=1e-12):
        if len(self.sectors) != self.L + 1:
            raise ValueError("Need L+1 sector arrays")

        for i, A in enumerate(self.tensors):
            qL, qR = self.sectors[i:i+2]

            if A.shape[0] != len(qL) or A.shape[2] != len(qR):
                raise ValueError("Sector dimensions mismatch")

            for a, left in enumerate(qL):
                for n in range(2):
                    for b, right in enumerate(qR):
                        allowed = (
                            left * n == 0
                            and (right == n or right == -1)
                        )
                        if not allowed and abs(A[a, n, b]) > atol:
                            raise ValueError(
                                f"Forbidden entry at site {i}"
                            )
        return True


def constrained_random_mps(L, seed=42):
    """Random admissible MPS with both sectors on internal bonds."""

    if L < 2:
        raise ValueError("L must be >= 2")

    rng = np.random.default_rng(seed)

    sectors = (
        [np.array([0])]
        + [np.array([0, 1]) for _ in range(L - 1)]
        + [np.array([-1])]
    )

    tensors = []

    for i in range(L):
        qL, qR = sectors[i:i+2]
        A = np.zeros((len(qL), 2, len(qR)))

        for a, left in enumerate(qL):
            for n in range(2):
                for b, right in enumerate(qR):
                    if left * n == 0 and (right == n or right == -1):
                        A[a, n, b] = rng.normal()

        tensors.append(A)

    mps = ConstrainedMPS(tensors, sectors)

    # Normalise without constructing the full wavefunction
    env = np.ones((1, 1))

    for A in mps.tensors:
        env = np.einsum(
            'ab,asr,bsu->ru',
            env, A.conj(), A,
            optimize=True
        )

    norm = np.sqrt(env[0, 0].real)
    if norm == 0:
        raise ValueError("Initial MPS has zero norm")

    mps.tensors[0] /= norm

    return mps
