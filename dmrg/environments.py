import numpy as np

# Want left_env, right_env = build_environments(mps, mpo, centre)

def build_environments(mps, mpo, centre):
    """ Centre is the index of the first site of the two-site cell"""

    left_range = np.arange(1, centre)
    right_range = np.arange(centre + 2, mps.L - 1)
    right_range = right_range[::-1] # Reverse order
    
    if centre == 0:
        L = np.ones((1, 1, 1))
    else:

        # Contract left

        A = mps.tensors[0]
        W = mpo.mpo_tensors[0]

        L = np.tensordot(A, W, axes=([1], [1]))

        A_L, A_R, W_L, d_out, W_R = L.shape
        L = np.transpose(L, (0, 2, 3, 1, 4)) # Eg new axis 1 from old axis 2 (0, 2, ...)
        # (A_L, W_L, d_out, A_R, W_R)

        L = np.tensordot(L, np.conj(A), axes=([2], [1]))
        A_L, W_L, A_R, W_R, A_bra_L, A_bra_R = L.shape
        L = np.transpose(L, (0, 1, 4, 2, 3, 5)) # (A_L, W_L, A_bra_L, A_R, W_R, A_bra_R)
        L = L[0, 0, 0, :, :, :] # L[0, # choose the only index, : # keep ALL indices]
        # (A_R, W_R, A_bra_R)

        for i in left_range:
            local = np.tensordot(mps.tensors[i], mpo.mpo_tensors[i], axes=(1, 1))
            # (A_L, A_R, W_L, d_out, W_R)
            local = np.tensordot(local, np.conj(mps.tensors[i]),  axes=(3, 1))
            #(A_L, A_R, W_L, W_R, A_bra_L, A_bra_R)

            L = np.tensordot(L, local, axes=(0, 0)) # (W_R, A_bra_R, A_R, W_L, W_R, A_bra_L, A_bra_R)
            L = np.einsum('abcaebf->cef', L)


    # Contract Right
    if centre + 2 == mps.L:
        R = np.ones((1, 1, 1))

    else:
        A = mps.tensors[-1]
        W = mpo.mpo_tensors[-1]

        R = np.tensordot(A, W, axes=([1], [1]))

        A_L, A_R, W_L, d_out, W_R = R.shape
        R = np.transpose(R, (0, 2, 3, 1, 4)) # Eg new axis 1 from old axis 2 (0, 2, ...)
        # (A_L, W_L, d_out, A_R, W_R)

        R = np.tensordot(R, np.conj(A), axes=([2], [1]))
        A_L, W_L, A_R, W_R, A_bra_L, A_bra_R = R.shape
        R = np.transpose(R, (0, 1, 4, 2, 3, 5)) # (A_L, W_L, A_bra_L, A_R, W_R, A_bra_R)
        R = R[:, :, :, 0, 0, 0] # R[0, # choose the only index, : # keep ALL indices]
        # (A_L, W_L, A_bra_L)

        for i in right_range:
            local = np.tensordot(mps.tensors[i], mpo.mpo_tensors[i], axes=(1, 1))
            # (A_L, A_R, W_L, d_out, W_R)
            local = np.tensordot(local, np.conj(mps.tensors[i]),  axes=(3, 1))
            #(A_L, A_R, W_L, W_R, A_bra_L, A_bra_R)

            R = np.tensordot(R, local, axes=(0, 1)) # (W_L, A_bra_L, A_L, W_L, W_R, A_bra_L, A_bra_R)
            # (W_L_old, A_bra_L_old, A_L_new, W_L_new, W_R_new, A_bra_L_new, A_bra_R_new)

            R = np.einsum('abceafb->cef', R)

    return L, R


def update_left_env(L, A, W):

    # L: (chi_L, D_L, chi_L)
    # A: (chi_L, d, chi_R)
    # W: (D_L, d_in, d_out, D_R)

    T = np.tensordot(L, A, axes=([0], [0]))
    # (D_L, chi_L, d_in, chi_R)

    T = np.tensordot(T, W, axes=([0, 2], [0, 1]))
    # (chi_L, chi_R, d_out, D_R)

    L_new = np.tensordot(
        T, A.conj(), axes=([0, 2], [0, 1])
    )
    # (chi_R, D_R, chi_R)

    return L_new


def update_right_env(R, A, W):

    # Contract ket with right environment
    T = np.tensordot(A, R, axes=([2], [0]))
    # (chi_L, d_in, D_R, chi_R)

    # Contract MPO
    T = np.tensordot(T, W, axes=([1, 2], [1, 3]))
    # (chi_L, chi_R, D_L, d_out)

    # Contract bra
    R_new = np.tensordot(
        T, A.conj(), axes=([1, 3], [2, 1])
    )
    # (chi_L, D_L, chi_L)

    return R_new
