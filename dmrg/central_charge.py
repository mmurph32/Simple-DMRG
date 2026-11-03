
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


import numpy as np


def fit_central_charge(S, trim=5):
    """
    Calabrese-Cardy fit for an open chain.
    No smoothing or assumed oscillation period.
    """

    S = np.asarray(S)
    L = len(S) + 1

    ell = np.arange(1, L)

    x = np.log(
        (2 * L / np.pi) *
        np.sin(np.pi * ell / L)
    )

    mask = (ell > trim) & (ell < L - trim)

    slope, intercept = np.polyfit(
        x[mask], S[mask], 1
    )

    c = 6 * slope

    return c, slope, intercept, x
