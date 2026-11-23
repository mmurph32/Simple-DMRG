import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from AGS_Anyon_Model.anyon_idmrg import anyon_grow_and_sweep
from dmrg.entanglement import entanglement_entropy

L = 30
chi = 32
jeff = -1.3

for x_final in (1, 0):

    print(f"\nTesting x_final = {x_final}")

    mps, E, E_before, history = anyon_grow_and_sweep(
        jeff=jeff,
        chi_max=chi,
        max_sites=L,
        penalty=75,
        number_of_sweeps=3,
        x_final=x_final
    )

    S = np.array(entanglement_entropy(mps))

    print("Energy before polishing:", E_before)
    print("Energy after polishing:", E)
    print("Maximum entropy:", np.max(S))
    print("Entropies:", np.round(S, 5))