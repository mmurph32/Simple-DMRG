import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from AGS_Anyon_Model.anyon_idmrg import anyon_grow_and_sweep
from dmrg.entanglement import entanglement_entropy

chi = 80
L = 99
Jeff = -1.0

mps, E, E_before, history = anyon_grow_and_sweep(
    jeff=Jeff,
    chi_max=chi,
    max_sites=L,
    penalty=100,
    number_of_sweeps=5,
    tol=1e-10,
    verbose=True,
    x_final=0 # (0, 1) = (1, tau)
)

# Entanglement entropy
S = entanglement_entropy(mps)

np.save(f"anyon_entropy_L={L}_chi={chi}_Jeff={Jeff}.npy", S)

print("Entropy saved.")

