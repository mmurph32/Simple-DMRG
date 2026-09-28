import numpy as np
from dmrg.environments import build_environments

def effective_action(mps, mpo, centre, theta):

    L, R = build_environments(mps, mpo, centre)

    # A_1 = mps.tensors[centre]
    # A_2 = mps.tensors[centre+1]

    W_1 = mpo.mpo_tensors[centre]
    W_2 = mpo.mpo_tensors[centre+1]

    # Build H_eff

    W_centre = np.tensordot(W_1, W_2, axes=(-1, 0)) # (W1_L, d_in1, dout1, din2, dout2, W2_R)
    
    H_eff = np.tensordot(L, W_centre, axes=(1, 0)) # (L_a_R, L_a_bra_R, d_in1, dout1, din2, dout2, W2_R)

    H_eff = np.tensordot(H_eff, R, axes=(-1, 1)) # (L_a_R, L_a_bra_R, d_in1, dout1, din2, dout2, R_a_L, R_a_bra_L)

    # Contract H_eff and Theta

    result = np.tensordot(theta, H_eff, axes=([1, 2], [2, 4])) # (A1_L, A2_R, L_a_R, L_a_bra_R, dout1, dout2, R_a_L, R_a_bra_L)

    # Contract Ket Virtual bomds

    result = np.einsum('abadefbh->defh', result) # final H_eff Theta = (L_a_bra_r, dout1, dout2, R_a_bra_L) and L_a_bra_r = AA_R
    return result