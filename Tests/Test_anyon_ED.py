
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from AGS_Anyon_Model.fibonacci import local_projector
from AGS_Anyon_Model.compressed import fibonacci_small_mpo

L = 8
jeff = 1.0
penalty = 50
x_final = 0

# Build dense golden-chain Hamiltonian independently
P = local_projector()
H = np.zeros((2**L, 2**L), dtype=complex)

for j in range(L - 2):
    H += jeff * np.kron(
        np.kron(np.eye(2**j), P),
        np.eye(2**(L-j-3))
    )

# Forbidden adjacent vacuum labels
for j in range(L - 1):
    forbidden = np.diag([1, 0, 0, 0])
    H += penalty * np.kron(
        np.kron(np.eye(2**j), forbidden),
        np.eye(2**(L-j-2))
    )

# Left boundary: x_0 = vacuum
H += penalty * np.kron(
    np.diag([0, 1]), np.eye(2**(L-1))
)

# Right boundary: x_final = vacuum
H += penalty * np.kron(
    np.eye(2**(L-1)), np.diag([0, 1])
)

# Contract the compressed MPO into a dense matrix
mpo = fibonacci_small_mpo(
    L, jeff, penalty=penalty, x_final=x_final
)

T = mpo.mpo_tensors[0]

for W in mpo.mpo_tensors[1:]:
    T = np.tensordot(T, W, axes=([-1], [0]))

# T has indices (1, s0, t0, s1, t1, ..., 1)
T = T.reshape((2, 2) * L)

# Group all input and output indices
axes = list(range(0, 2*L, 2)) + list(range(1, 2*L, 2))
H_mpo = T.transpose(axes).reshape(2**L, 2**L)

error = np.max(np.abs(H_mpo - H))

print("Maximum MPO error:", error)
print("MPO Hermiticity error:",
      np.max(np.abs(H_mpo - H_mpo.conj().T)))
