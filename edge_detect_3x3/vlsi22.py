from __future__ import annotations
import numpy as np
import math

def vlsi22_generate_stream_for_value(p_prob: float, lengthN: int) -> np.ndarray:
    """
    Deterministic GB stream of length P=2^n that must be a perfect square: P=q^2.
    """
    P = int(lengthN)
    q = int(round(np.sqrt(P)))
    assert q * q == P, "VLSI22/Downscale requires lengthN to be a perfect square (e.g., 256, 1024)"

    k = int(round(float(p_prob) * P))
    k = max(0, min(P, k))

    A_q = k // q
    err = k - A_q * q

    out = np.zeros(P, dtype=np.uint8)
    if A_q > 0:
        out[:A_q * q] = 1

    if err > 0 and A_q < q:
        positions = np.linspace(0, q - 1, err, dtype=int)
        row_start = A_q * q
        for pos in positions:
            out[row_start + pos] = 1

    return out

def vlsi22_pair_streams(p_prob: float, w_prob: float, P: int):
    """
    Pair streams with row-repeat (A) and column-tile (B), plus compensation.
    """
    assert int(np.sqrt(P))**2 == P, "VLSI22 requires P to be a perfect square"
    q = int(np.sqrt(P))

    def _downscale_tb_local(p):
        k = int(round(float(p) * P))
        A_q = k // q
        e   = k - A_q * q
        tb  = np.zeros(q, dtype=np.uint8)
        if A_q > 0:
            tb[:A_q] = 1
        return tb, A_q, e

    tbA, A_q, eA = _downscale_tb_local(p_prob)
    tbB, B_q, eB = _downscale_tb_local(w_prob)

    sA = np.repeat(tbA, q)
    sB = np.tile(tbB, q)

    if (A_q < q) and (eA > 0):
        invA = int(round(eA * (B_q / q)))
        invA = max(0, min(invA, q))
        if invA > 0:
            cols_one = np.flatnonzero(tbB == 1)
            if cols_one.size > 0:
                pick = np.linspace(0, cols_one.size - 1, invA, dtype=int)
                sA[A_q * q + cols_one[pick]] = 1
            else:
                pick = np.linspace(0, q - 1, invA, dtype=int)
                sA[A_q * q + pick] = 1

    if (B_q < q) and (eB > 0):
        invB = int(round(eB * (A_q / q)))
        invB = max(0, min(invB, q))
        if invB > 0:
            rows_one = np.flatnonzero(tbA == 1)
            if rows_one.size > 0:
                pick = np.linspace(0, rows_one.size - 1, invB, dtype=int)
                rsel = rows_one[pick]
                sB[rsel * q + B_q] = 1
            else:
                pick = np.linspace(0, q - 1, invB, dtype=int)
                sB[pick * q + B_q] = 1

    return sA.astype(np.uint8), sB.astype(np.uint8)