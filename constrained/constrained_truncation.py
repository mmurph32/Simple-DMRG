
import numpy as np


def constrained_truncation(
    theta, qL, qR, chi_max, direction, svd_min=1e-12
):
    """
    Sector-preserving two-site SVD.

    theta[a, n1, n2, b]
    qL[a] = left virtual sector
    qR[b] = right virtual sector
    """

    chiL, d1, d2, chiR = theta.shape

    blocks = []

    for q in (0, 1):

        # Left basis: (a, n1), with internal sector q=n1
        left_indices = [
            (a, n1)
            for a in range(chiL)
            for n1 in range(d1)
            if n1 == q and qL[a] * n1 == 0
        ]

        # Right basis: (n2, b), compatible with q
        right_indices = [
            (n2, b)
            for n2 in range(d2)
            for b in range(chiR)
            if (qR[b] == n2 or qR[b] == -1) and q * n2 == 0
        ]

        if not left_indices or not right_indices:
            continue

        M = np.array([
            [theta[a, n1, n2, b] for n2, b in right_indices]
            for a, n1 in left_indices
        ])

        U, S, Vh = np.linalg.svd(M, full_matrices=False)

        blocks.append((q, left_indices, right_indices, U, S, Vh))

    # Collect singular values from both sectors
    candidates = []

    for block_id, (_, _, _, U, S, Vh) in enumerate(blocks):
        for j, value in enumerate(S):
            candidates.append((value, block_id, j))

    if not candidates:
        raise ValueError("No allowed Schmidt sectors")

    candidates.sort(reverse=True)

    norm = np.sqrt(sum(value**2 for value, _, _ in candidates))

    selected = [
        (block_id, j)
        for value, block_id, j in candidates
        if value / norm >= svd_min
    ][:chi_max]

    if not selected:
        selected = [(candidates[0][1], candidates[0][2])]

    chi = len(selected)

    A = np.zeros((chiL, d1, chi), dtype=theta.dtype)
    B = np.zeros((chi, d2, chiR), dtype=theta.dtype)
    q_mid = np.zeros(chi, dtype=int)

    retained = np.array([
        blocks[block_id][4][j]
        for block_id, j in selected
    ])

    retained /= np.linalg.norm(retained)

    for k, (block_id, j) in enumerate(selected):

        q, left_idx, right_idx, U, S, Vh = blocks[block_id]
        q_mid[k] = q

        for row, (a, n1) in enumerate(left_idx):
            A[a, n1, k] = U[row, j]

        for col, (n2, b) in enumerate(right_idx):
            B[k, n2, b] = Vh[j, col]

    if direction == "right":
        B *= retained[:, None, None]

    elif direction == "left":
        A *= retained[None, None, :]

    else:
        raise ValueError("direction must be 'right' or 'left'")

    return A, B, q_mid
