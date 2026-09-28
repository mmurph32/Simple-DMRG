import numpy as np
from dmrg.eigensolver import eigensolver
from dmrg.truncation import truncation
from dmrg.energy import energy

def sweep(mps, mpo, bond_dim, number_of_sweeps, tol):

    E_prev = energy(mps, mpo)

    for n in range(number_of_sweeps):

        # Sweep left -> right
        for i in range(0, mps.L - 1):

            theta = np.tensordot(
                mps.tensors[i],
                mps.tensors[i + 1],
                axes=(-1, 0)
            )

            e_val, theta_new = eigensolver(mps, mpo, i, theta)

            A, B = truncation(theta_new, bond_dim, "right")

            mps.tensors[i] = A
            mps.tensors[i + 1] = B


        # Sweep right -> left
        for i in range(1, mps.L)[::-1]:

            theta = np.tensordot(
                mps.tensors[i - 1],
                mps.tensors[i],
                axes=(-1, 0)
            )

            e_val, theta_new = eigensolver(mps, mpo, i - 1, theta)

            A, B = truncation(theta_new, bond_dim, "left")

            mps.tensors[i - 1] = A
            mps.tensors[i] = B

        E_new = energy(mps, mpo)
        print("Sweep", n+1, "finsihed.")

        if abs(E_new - E_prev) < tol:
            print("Tolerance reached after", n+1, "sweeps.")
            break
        else:
            E_prev = E_new


    return mps, E_new