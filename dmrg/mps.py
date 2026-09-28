import numpy as np 

class SimpleMPS: 
    """class for a matrix product state
    with physical dimension d, right and left bond dimension chi_l and chi_R, and length L
    
    The tensors are of the shape (chi_L, d, chi_R)
    """
    def __init__(self, tensors):
        self.tensors = tensors
        self.L = len(tensors)
        self.d = tensors[0].shape[1]

    def check_valid(self):
        if self.tensors[0].shape[0] != 1 or self.tensors[-1].shape[2] != 1:
            raise ValueError("Edge bond dimensions incorrect")

        for tensor in self.tensors:
            if tensor.shape[1] != self.d:
                raise ValueError("Physical dimension d is not uniform")
            if tensor.ndim != 3:
                raise ValueError("Tensor shape is not uniform")

        for i in range(len(self.tensors) - 1):
            if self.tensors[i].shape[2] != self.tensors[i+1].shape[0]:
                raise ValueError("Tensors' indices don't match")

        print("MPS is valid")
        return

    def contract(self):
        result = self.tensors[0]

        for i in range(1, self.L):
            result = np.tensordot(result, self.tensors[i], axes=[-1, 0])

        result = np.squeeze(result)
        result = result.reshape(self.d**self.L)
        return result

def product_state(phi, L):
    # create L tensors, each with shape (1, d, 1)
    # put phi inside each tensor eg. phi = [1, 0]
    # return an MPS made from those tensors
    tensors = []
    d = len(phi)

    for i in range(L):
        state = np.zeros((1, d, 1))
        state[0, :, 0] = phi
        tensors.append(state)

    return SimpleMPS(tensors)
