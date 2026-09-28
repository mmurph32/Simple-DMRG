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