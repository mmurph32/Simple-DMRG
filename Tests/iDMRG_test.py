import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmrg.idmrg import idmrg
from dmrg.sweep import sweep
from dmrg.tfim import tfim_mpo
from dmrg.energy import energy
from dmrg.eigensolver import right_canonicalize

J = 1.0
h = 1.0
chi = 32

# Stage 1: Grow the system
mps, history = idmrg(
    mpo3=tfim_mpo(3, J, h),
    phi=[1.0, 0.0],
    chi_max=chi,
    max_sites=100,
    tol=None
)

# Stage 2: Prepare for finite sweeps
mpo = tfim_mpo(mps.L, J, h)
mps = right_canonicalize(mps)

E_before = energy(mps, mpo)

# Stage 3: Polish the state
mps, E_after = sweep(
    mps,
    mpo,
    bond_dim=chi,
    number_of_sweeps=20,
    tol=1e-10
)

print("Final length:", mps.L)
print("Energy before polishing:", E_before)
print("Energy after polishing: ", E_after)
print("Energy improvement:     ", E_before - E_after)