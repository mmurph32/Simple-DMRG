import numpy as np

def entanglement_entropy(input_mps):
    mps = [A.copy() for A in input_mps.tensors] # mps is not in the SimpleMPS class, it is just a list
    
    L = len(mps)

    # Step 1: Sweep right to left and orthogonalise using QR
    for i in range(L - 1, 0, -1):
        A = mps[i]

        chi_L, d, chi_R = A.shape
        A = A.reshape(chi_L, d * chi_R)

        Q, R = np.linalg.qr(A.conj().T) # Want Ortho rows, so we do A.T, 

        # Set Q.T = A_i, and absorb R.T into A_i-1
        
        new_chi_L = Q.shape[1] # new_chi_L = k
        mps[i] = Q.conj().T.reshape(new_chi_L, d, chi_R)

        mps[i-1] = np.tensordot(mps[i-1], R.conj().T, (-1, 0))

    # Step 2: Sweep left to right, calculating entanglement entropy at ecah site via SVD

    entropies = []

    for i in range(0, L-1):
        A = mps[i]
        chi_L, d, chi_R = A.shape
        A = A.reshape(chi_L * d, chi_R)
        U, S, V = np.linalg.svd(A, full_matrices=False) # Need A as a matrix

        p = S**2
        p = p / np.sum(p)  # Normalise

        p = p[p > 1e-15]  # Avoid log(0)

        entropy = -np.sum(p * np.log(p))
        entropies.append(entropy)

        new_chi_R = len(S)

        mps[i] = U.reshape(chi_L, d, new_chi_R)

        mps[i+1] = np.tensordot(
            np.diag(S) @ V,
            mps[i+1],
            axes=([-1], [0])
        )

    return np.array(entropies)
