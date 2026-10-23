import numpy as np

def effective_action_fast(mpo, centre, theta, L_env, R_env):

    W1 = mpo.mpo_tensors[centre]
    W2 = mpo.mpo_tensors[centre + 1]

    result = np.einsum(
    'awb,wsuv,vtxz,czd,astc->buxd',
    L_env, W1, W2, R_env, theta,
    optimize=False
    )


    return result

