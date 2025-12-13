from __future__ import annotations
import numpy as np
from scipy.stats.qmc import Sobol
from utils import is_pow2

def htc_build_fsm_schedule_sobol(lengthN: int) -> np.ndarray:
    """
    Build HTC FSM SEL[k] in [0..n-1] from 1D Sobol numbers.
    lengthN must be 2^n.
    """
    assert is_pow2(lengthN), "HTC requires lengthN=2^n"
    n = int(np.log2(lengthN))
    eng = Sobol(d=1, scramble=False, seed=None)
    S = eng.random(n=lengthN)[:, 0]
    SEL = np.empty(lengthN, dtype=np.int32)

    for k, Sk in enumerate(S):
        for m in range(1, n + 1):
            left  = (2**(m-1) - 1) / (2**(m-1)) if m > 1 else 0.0
            right = (2**m - 1) / (2**m)
            if left <= Sk < right:
                SEL[k] = m - 1
                break
        else:
            SEL[k] = n - 1
    return SEL

def htc_int_to_bits(x_int: int, n: int) -> np.ndarray:
    bits = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        bits[n - 1 - i] = (x_int >> i) & 1
    return bits

def htc_encode_RB_by_FSM(p_prob: float, lengthN: int, sel: np.ndarray) -> np.ndarray:
    assert len(sel) == lengthN and is_pow2(lengthN)
    n = int(np.log2(lengthN))
    x_int = int(round(p_prob * (2**n)))
    if x_int >= 2**n:
        x_int = 2**n - 1
    if x_int <= 0:
        x_bits = np.zeros(n, dtype=np.uint8)
    else:
        x_bits = htc_int_to_bits(x_int, n)

    out = np.empty(lengthN, dtype=np.uint8)
    for k in range(lengthN):
        out[k] = x_bits[sel[k]]
    return out

def htc_encode_TB(p_prob: float, lengthN: int) -> np.ndarray:
    val8 = int(round(p_prob * 255.0))
    if val8 <= 0:
        return np.zeros(lengthN, dtype=np.uint8)
    if val8 >= 255:
        return np.ones(lengthN, dtype=np.uint8)
    ones = int(round(val8 * lengthN / 255.0))
    ones = max(0, min(lengthN, ones))
    out = np.zeros(lengthN, dtype=np.uint8)
    out[:ones] = 1
    return out