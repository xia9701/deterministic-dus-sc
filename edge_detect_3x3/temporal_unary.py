from __future__ import annotations
import math
import numpy as np

def conv3x3_temporal_unary_counts(gray: np.ndarray, Kx: np.ndarray, Ky: np.ndarray, N: int) -> np.ndarray:
    """
    Generic 3x3 tubGEMM in count domain (matches conv3x3_temporal_unary_counts).
    Returns unnormalized |G| in bipolar scale, later normalized by norm_factor.
    """
    H, W = gray.shape
    pad = np.pad(gray, ((1,1),(1,1)), mode='reflect')
    out = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            gx_counts = 0
            gy_counts = 0

            for mat, assign in ((Kx, 'gx'), (Ky, 'gy')):
                total_counts = 0
                for m in range(3):
                    for n in range(3):
                        w = float(mat[m, n])
                        if w == 0.0:
                            continue
                        p = float(pad[i + m, j + n])
                        x = 2.0 * p - 1.0

                        xp = max(x, 0.0); xn = max(-x, 0.0)
                        wp = max(w, 0.0); wn = max(-w, 0.0)

                        Kxp = int(round(xp * N)); Kxn = int(round(xn * N))
                        Kwp = int(round(wp * N)); Kwn = int(round(wn * N))

                        Kpp = (Kxp * Kwp) // N
                        Knn = (Kxn * Kwn) // N
                        Kpn = (Kxp * Kwn) // N
                        Knp = (Kxn * Kwp) // N

                        Kprod_bip = (Kpp + Knn) - (Kpn + Knp)
                        total_counts += Kprod_bip

                if assign == 'gx':
                    gx_counts = total_counts
                else:
                    gy_counts = total_counts

            gx = gx_counts / float(N)
            gy = gy_counts / float(N)
            out[i, j] = math.sqrt(gx*gx + gy*gy)

    return out