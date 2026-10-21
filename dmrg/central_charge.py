
import numpy as np

def extract_central_charge(entropies):
    L = len(entropies) + 1

    L_range = np.arange(1, L)

    # Calabrese-Cardy conformal distance
    x = np.log(
        (2 * L / np.pi) * np.sin(np.pi * L_range / L)
    )

    # Fit S = slope*x + intercept
    slope, intercept = np.polyfit(x, entropies, 1)

    c = 6 * slope

    return c, slope, intercept


