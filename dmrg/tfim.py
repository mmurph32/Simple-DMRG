import numpy as np
from dmrg.mpo import SimpleMPO

identity = np.array([[1, 0],
                        [0, 1]])
X = np.array([[0,1],
                [1,0]])

Z = np.array([[1,0], 
                [0,-1]])

def tfim_mpo(L, J, h):

    # construct first_W
    first_W =  np.zeros((1, 2, 2, 3))
    first_W[0, :, :, 0] = identity
    first_W[0, :, :, 1] = Z
    first_W[0, :, :, 2] = -h * X


    # construct W
    W = np.zeros((3, 2, 2, 3))

    W[0, :, :, 0] = identity
    W[0, :, :, 1] = Z
    W[0, :, :, 2] = -h * X

    W[1, :, :, 2] = -J * Z
    W[2, :, :, 2] = identity

    # construct last_W
    last_W =  np.zeros((3, 2, 2, 1))
    last_W[0, :, :, 0] = -h*X
    last_W[1, :, :, 0] = -J * Z
    last_W[2, :, :, 0] = identity

    tensors = [first_W]

    # add L-2 copies of W
    for i in range(L-2):
        tensors.append(W)

    tensors.append(last_W)

    return SimpleMPO(tensors)