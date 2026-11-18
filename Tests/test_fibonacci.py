import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from AGS_Anyon_Model.anyon_idmrg import anyon_grow_and_sweep
from dmrg.entanglement import entanglement_entropy

mps, E, E_before, history = anyon_grow_and_sweep(
    jeff=-1.3,
    chi_max=32,
    max_sites=100,
    penalty=50,
    number_of_sweeps=2,
    tol=1e-7,
    verbose=True
)

# Entanglement entropy
S = entanglement_entropy(mps)

np.save("anyon_entropy.npy", S)

print("Entropy saved.")