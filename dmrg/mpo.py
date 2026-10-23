import numpy as np

class SimpleMPO:
    """MPO chain of operators with shape,
    [D_L, d_in, d_out, D_R]"""

    def __init__(self, mpo_tensors):
        self.mpo_tensors = mpo_tensors
        self.L = len(mpo_tensors)
        self.d_in = mpo_tensors[0].shape[1]
        self.d_out = mpo_tensors[0].shape[2]

    def check_valid(self):
        if self.mpo_tensors[0].shape[0] != 1 or self.mpo_tensors[-1].shape[-1] != 1:
            raise ValueError("MPO Edge bond dimensions incorrect")

        for tensor in self.mpo_tensors:
            if tensor.shape[1] != self.d_in:
                raise ValueError("MPO Physical dimension d_in is not uniform")
            if tensor.shape[2] != self.d_out:
                raise ValueError("MPO Physical dimension d_out is not uniform")
            if tensor.ndim != 4:
                raise ValueError("MPO is not rank 4")

        for i in range(len(self.mpo_tensors) - 1):
            if self.mpo_tensors[i].shape[-1] != self.mpo_tensors[i+1].shape[0]:
                raise ValueError("MPO tensors' indices don't match")

        print("MPO is valid")
        return 

    def contract(self):
        result = self.mpo_tensors[0]

        for i in range(1, self.L):
            result = np.tensordot(result, self.mpo_tensors[i], axes=[-1, 0])

        result = np.squeeze(result)

        # Reorder to d_in, d_in, d_out, d_out
        ket_axes = list(range(0, 2*self.L, 2)) 
        bra_axes = list(range(1, 2*self.L, 2))
        transpose_order = ket_axes + bra_axes

        result = result.transpose(transpose_order)
        result = result.reshape(2**self.L, 2**self.L)
        return result