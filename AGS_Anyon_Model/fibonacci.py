"""Fibonacci fusion-path projectors and Kirchner et al. effective Hamiltonian.

Local label convention: 0 = vacuum (1), 1 = tau.
Projector P_j acts on fusion labels (j,j+1,j+2).
Open-chain Hamiltonian includes all fully supported terms.
"""
import numpy as np

PHI = (1 + np.sqrt(5.0)) / 2


def local_projector():
    """8x8 matrix in computational order |000>,...,|111>.

    Physical sectors follow Trebst et al. Eq. (3.4):
    |010> -> 1, |011> -> 0, and the |101>,|111> sector
    carries the nontrivial 2x2 F^dagger |1><1| F block.
    """
    p = np.zeros((8, 8), dtype=float)
    p[2, 2] = 1.0          # |0,1,0>
    a, b = 1 / PHI, 1 / np.sqrt(PHI)
    p[np.ix_([5, 7], [5, 7])] = np.outer([a, b], [a, b])
    return p


def allowed(bits):
    return all(not (a == 0 and b == 0) for a, b in zip(bits[:-1], bits[1:]))


def projector_dense(L, j):
    """Pair-fusion projector on label positions j,j+1,j+2 (zero based)."""
    if not (0 <= j <= L - 3):
        raise ValueError('projector support outside chain')
    eye = np.eye(2 ** j)
    tail = np.eye(2 ** (L - j - 3))
    return np.kron(np.kron(eye, local_projector()), tail)


def hamiltonian_dense(L, jeff, penalty=0.0, theta=6*np.pi/5):
    """Finite open-chain benchmark of Eq. (9), with j=0 as even bond.

    H = jeff * sum_j P_j - sum_j [alpha_j P_j P_(j+1) + h.c.]
    alpha_j = exp(-i theta) for even j, exp(+i theta) for odd j.
    """
    if L < 3:
        raise ValueError('L must be >= 3')
    projectors = [projector_dense(L, j) for j in range(L - 2)]
    H = np.zeros((2**L, 2**L), dtype=complex)
    for P in projectors:
        H += jeff * P
    for j in range(len(projectors)-1):
        alpha = np.exp((-1 if j % 2 == 0 else 1) * 1j * theta)
        product = projectors[j] @ projectors[j+1]
        H -= alpha * product + alpha.conjugate() * product.conj().T
    if penalty:
        for state in range(2**L):
            bits = [(state >> (L-1-k)) & 1 for k in range(L)]
            H[state, state] += penalty * sum(
                a == 0 and b == 0 for a, b in zip(bits[:-1], bits[1:])
            )
    return H


def _operator_tt(matrix, sites, cutoff=1e-13):
    """Factor a k-site operator into [left, input, output, right] MPO tensors."""
    k = sites
    tensor = matrix.reshape((2,) * k * 2).transpose(
        [axis for pair in zip(range(k), range(k, 2*k)) for axis in pair]
    )
    tensors = []
    rank = 1
    for _ in range(k-1):
        mat = tensor.reshape(rank*4, -1)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        keep = max(1, np.count_nonzero(s > cutoff*s[0]))
        tensors.append(u[:, :keep].reshape(rank, 2, 2, keep))
        tensor = s[:keep, None]*vh[:keep]
        rank = keep
    tensors.append(tensor.reshape(rank, 2, 2, 1))
    return tensors


def _term_mpo(L, start, local_matrix):
    k = int(round(np.log2(local_matrix.shape[0])))
    pieces = _operator_tt(local_matrix, k)
    identity = np.eye(2, dtype=complex).reshape(1, 2, 2, 1)
    return [identity.copy() for _ in range(start)] + pieces + [
        identity.copy() for _ in range(L-start-k)]


def _sum_mpos(mpos):
    """Exact direct-sum construction (reference MPO, not bond optimized)."""
    L = len(mpos[0])
    if L == 1:
        return [sum((m[0] for m in mpos))]
    result = []
    for i in range(L):
        left = 1 if i == 0 else sum(m[i].shape[0] for m in mpos)
        right = 1 if i == L-1 else sum(m[i].shape[3] for m in mpos)
        W = np.zeros((left, 2, 2, right), dtype=complex)
        a = b = 0
        for m in mpos:
            t = m[i]
            la, rb = t.shape[0], t.shape[3]
            W[a:a+la, :, :, b:b+rb] += t
            if i != 0: a += la
            if i != L-1: b += rb
        result.append(W)
    return result


def fibonacci_mpo(L, jeff, penalty=75.0, theta=6*np.pi/5):
    """Exact finite-chain MPO matching hamiltonian_dense(L,...).

    This reference implementation uses a direct sum of local-term MPOs;
    its bond dimension grows with L. 
    """
    from dmrg.mpo import SimpleMPO
    if L < 4:
        raise ValueError('L must be >= 4')
    P = local_projector().astype(complex)
    I = np.eye(2, dtype=complex)
    Pleft = np.kron(P, I)
    Pright = np.kron(I, P)
    terms = []
    for j in range(L-2):
        terms.append(_term_mpo(L, j, jeff*P))
    for j in range(L-3):
        alpha = np.exp(((-1)**(j+1))*1j*theta)
        product = Pleft @ Pright
        ring = -(alpha*product + alpha.conjugate()*product.conj().T)
        terms.append(_term_mpo(L, j, ring))
    if penalty:
        forbidden = np.diag([1, 0, 0, 0]).astype(complex)
        for j in range(L-1):
            terms.append(_term_mpo(L, j, penalty*forbidden))
    return SimpleMPO(_sum_mpos(terms))


def fibonacci_bulk(jeff, penalty=75.0, theta=6*np.pi/5,
                   x_final=1, boundary_penalty=None):
    """Constant-bond MPO automaton. Returns (left, even, odd, right).

    Site labels are fusion-path labels (0=vacuum, 1=tau).
    Local P_j acts on labels j,j+1,j+2.  The ring term starts
    at j and alternates with j parity.  Open boundaries are obtained
    by taking the first row and final column of the bulk tensors.
    """
    P = local_projector().astype(complex)
    I = np.eye(2, dtype=complex)
    P0 = np.kron(P, I)
    P1 = np.kron(I, P)
    forbidden = np.diag([1, 0, 0, 0]).astype(complex)

    # Each term has its own finite number of in-flight channels.
    # There are no channels proportional to chain length.
    terms = []
    if jeff != 0:
        terms.append((None, _operator_tt(jeff*P, 3)))
    if penalty != 0:
        terms.append((None, _operator_tt(penalty*forbidden, 2)))
    # for parity in (0, 1):
    #     alpha = np.exp(((-1)**(parity+1))*1j*theta)
    #     product = P0 @ P1
    #     ring = -(alpha*product + alpha.conjugate()*product.conj().T) *0 #TEST  H_ring = 0 -> GC Hamiltonian
    #     terms.append((parity, _operator_tt(ring, 4)))

    # channel 0: ready to start a term; channel 1: term finished.
    # Interior channels carry the TT bonds of local operators.
    D = 2
    offsets = []
    for parity, pieces in terms:
        offsets.append(D)
        D += sum(w.shape[-1] for w in pieces[:-1])

    W = [np.zeros((D, 2, 2, D), dtype=complex) for _ in range(2)]
    for w in W:
        w[0, :, :, 0] = I
        w[1, :, :, 1] = I
    for (parity, pieces), offset in zip(terms, offsets):
        channels = []
        idx = offset
        for piece in pieces[:-1]:
            channels.append(np.arange(idx, idx+piece.shape[-1]))
            idx += piece.shape[-1]
        for start_parity in (0, 1):
            if parity is not None and parity != start_parity:
                continue
            W[start_parity][np.ix_([0], range(2), range(2), channels[0])] += pieces[0]
        for k in range(1, len(pieces)-1):
            for site_parity in (0, 1):
                # ring terms only start at their prescribed parity
                if parity is not None and site_parity != (parity+k)%2:
                    continue
                W[site_parity][np.ix_(channels[k-1], range(2), range(2), channels[k])] += pieces[k]
        for site_parity in (0, 1):
            if parity is not None and site_parity != (parity+len(pieces)-1)%2:
                continue
            W[site_parity][np.ix_(channels[-1], range(2), range(2), [1])] += pieces[-1]

    # Boundary charges: x_0 = vacuum (0), x_{L-1} = x_final.
    # Boundary penalties are applied only to the first and last sites.
    if x_final not in (0, 1):
         raise ValueError('x_final must be 0 (vacuum) or 1 (tau)')
    bp = penalty if boundary_penalty is None else boundary_penalty
    first = W[0][0:1].copy()
    last = W[1][:, :, :, 1:2].copy()
    first[0, :, :, 1] += bp * np.diag([0, 1])
    last[0, :, :, 0] += bp * np.diag([int(x_final == 1),
                                       int(x_final == 0)])
    return first, W[0], W[1], last


def fibonacci_scalable_mpo(L, jeff, penalty=75.0, theta=6*np.pi/5,
                           x_final=1, boundary_penalty=None):
    """Open-chain fixed-bond MPO, even L >= 4."""
    from dmrg.mpo import SimpleMPO
    if L < 4 or L % 2:
        raise ValueError('L must be even and >= 4')
    first, even, odd, last = fibonacci_bulk(jeff, penalty, theta,
                                          x_final, boundary_penalty)
    return SimpleMPO([first] + [even if j%2 == 0 else odd
                                for j in range(1, L-1)] + [last])
