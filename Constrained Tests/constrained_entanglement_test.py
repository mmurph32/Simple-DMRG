
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from dmrg.constrained_mps import constrained_random_mps
from dmrg.constrained_mpo import constrained_hard_boson_mpo
from dmrg.constrained_hard_boson import constrained_hamiltonian
from dmrg.constrained_sweep import constrained_sweep
from dmrg.entanglement import entanglement_entropy


# Parameters
L = 16
w = 1.0
U = -3.0299
V = 3.3302
chi = 64


# 1. Exact diagonalisation
H, states = constrained_hamiltonian(L, w, U, V)

E_exact, vectors = eigsh(
    H, k=1, which="SA", tol=1e-12
)

# Embed constrained state into full Hilbert space
psi_exact = np.zeros(2**L)

for j, state in enumerate(states):
    psi_exact[state] = vectors[j, 0]

# Exact entanglement entropy
S_exact = []

for ell in range(1, L):

    M = psi_exact.reshape(2**ell, 2**(L-ell))
    s = np.linalg.svd(M, compute_uv=False)

    p = s**2
    p = p[p > 1e-15]

    S_exact.append(-np.sum(p * np.log(p)))

S_exact = np.array(S_exact)


# 2. Constrained DMRG
mpo = constrained_hard_boson_mpo(L, w, U, V)
mps = constrained_random_mps(L)

mps, E_dmrg = constrained_sweep(
    mps, mpo,
    bond_dim=chi,
    number_of_sweeps=20,
    tol=1e-10,
    verbose=False
)

S_dmrg = entanglement_entropy(mps)


# 3. Compare
print("Exact energy:", E_exact[0])
print("DMRG energy: ", E_dmrg)
print("Energy error:", abs(E_dmrg - E_exact[0]))

print("\nMaximum entropy difference:",
      np.max(np.abs(S_exact - S_dmrg)))


plt.figure()

plt.plot(S_exact, "o-", label="Exact diagonalisation")
plt.plot(S_dmrg, "x--", label="Constrained DMRG")

plt.xlabel("Bond position")
plt.ylabel("Entanglement entropy")
plt.legend()
plt.show()
