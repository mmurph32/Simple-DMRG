"""Small-chain ED benchmark for the current AGS bulk-label convention.
Place in Tests/test_anyon_ed_simple.py and run from the project root.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PHI = (1 + np.sqrt(5)) / 2

def projector():
    p = np.zeros((8, 8), complex)
    p[2, 2] = 1
    v = np.array([1 / PHI, 1 / np.sqrt(PHI)])
    p[np.ix_([5, 7], [5, 7])] = np.outer(v, v)
    return p

def allowed_states(L, x_final=1):
    return [n for n in range(2**L) if (n >> (L-1)) == 0
            and (n & 1) == x_final and all(
        ((n >> (L - 2 - j)) & 3) != 0 for j in range(L - 1))]

def hamiltonian(L, jeff, x_final=1):
    states = allowed_states(L, x_final)
    lookup = {n: i for i, n in enumerate(states)}
    p = projector()
    q = np.kron(p, np.eye(2)) @ np.kron(np.eye(2), p)
    theta = 6 * np.pi / 5
    ring = [-(np.exp((-1 if parity == 0 else 1)*1j*theta)*q +
              np.exp((1 if parity == 0 else -1)*1j*theta)*q.conj().T)
            for parity in (0, 1)]
    rows, cols, vals = [], [], []
    for n in states:
        for pos, op in ([(j, jeff*p) for j in range(L-2)] +
                        [(j, ring[j % 2]) for j in range(L-3)]):
            k = int(np.log2(op.shape[0]))
            shift = L - pos - k
            mask = (1 << k) - 1
            old = (n >> shift) & mask
            for new in range(mask+1):
                amp = op[new, old]
                if abs(amp) < 1e-13:
                    continue
                target = (n & ~(mask << shift)) | (new << shift)
                if target in lookup:
                    rows.append(lookup[target])
                    cols.append(lookup[n])
                    vals.append(amp)
    H = coo_matrix((vals, (rows, cols)), shape=(len(states),)*2).tocsr()
    return H, states

def entropies(psi, states, L):
    full = np.zeros(2**L, complex)
    full[states] = psi
    result = []
    for cut in range(1, L):
        s = np.linalg.svd(full.reshape(2**cut, -1), compute_uv=False)
        probs = s*s
        probs = probs[probs > 1e-14]
        result.append(-np.sum(probs*np.log(probs)))
    return np.array(result)

if __name__ == '__main__':
    from AGS_Anyon_Model.anyon_idmrg import anyon_grow_and_sweep
    from AGS_Anyon_Model.fibonacci import fibonacci_scalable_mpo
    from dmrg.energy import energy

    L, jeff, penalty = 10, -1.3, 75
    for x_final in (1, 0):
        H, states = hamiltonian(L, jeff, x_final)
        eigenvalues, vecs = eigsh(H, k=1, which='SA', tol=1e-11)
        E_ed = eigenvalues[0]
        print(f"x_0=0, x_final={x_final}, states={len(states)}, ED={E_ed:.12f}")
        mps, E, _, _ = anyon_grow_and_sweep(
            jeff, chi_max=32, max_sites=L, penalty=penalty,
            number_of_sweeps=3, x_final=x_final)
        mpo = fibonacci_scalable_mpo(L, jeff, penalty, x_final=x_final)
        E_dmrg = energy(mps, mpo)
        print(f"DMRG={E_dmrg:.12f}, error={abs(E_dmrg-E_ed):.3e}")
        print('ED entropy:', np.round(entropies(vecs[:, 0], states, L), 6))


from dmrg.entanglement import entanglement_entropy

S_ed = entropies(vecs[:, 0], states, L)
S_dmrg = np.array(entanglement_entropy(mps))

print("DMRG entropy:", np.round(S_dmrg, 6))
print("Maximum entropy error:",
      np.max(np.abs(S_dmrg - S_ed)))