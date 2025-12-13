# baselines.py
# Random / LFSR / Halton / Sobol / uMUL baselines.

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats.qmc import Halton, Sobol

from utils import generate_random_nbit_nonzero, normalize_to_threshold_space


# -------------------------
# Random baseline (per-bit fresh random n-bit threshold)
# -------------------------

def random_bitstream(binary_str: str, lengthN: int, bit_width: int) -> str:
    """
    For each unary bit, generate a fresh random n-bit threshold and compare.
    """
    x = int(binary_str, 2)
    out = []
    for _ in range(lengthN):
        thr = int(generate_random_nbit_nonzero(bit_width), 2)
        out.append("1" if x > thr else "0")
    return "".join(out)


# -------------------------
# Halton / Sobol threshold arrays
# -------------------------

def halton_thresholds(lengthN: int) -> Tuple[List[int], List[int]]:
    """
      halton_engine = Halton(d=2, scramble=False)
      engine.random(n=N) twice
    """
    engine = Halton(d=2, scramble=False)
    s1 = engine.random(n=lengthN)
    s2 = engine.random(n=lengthN)
    H1 = (s1[:, 0] * lengthN).astype(int).tolist()
    H2 = (s2[:, 1] * lengthN).astype(int).tolist()
    return H1, H2


def sobol_thresholds(lengthN: int) -> Tuple[List[int], List[int]]:
    """
      sobol_engine = Sobol(d=2, scramble=False)
      engine.random(n=N) twice
    """
    engine = Sobol(d=2, scramble=False)
    s1 = engine.random(n=lengthN)
    s2 = engine.random(n=lengthN)
    S1 = (s1[:, 0] * lengthN).astype(int).tolist()
    S2 = (s2[:, 1] * lengthN).astype(int).tolist()
    return S1, S2


def stream_from_thresholds(binary_str: str, thresholds: List[int], lengthN: int, max_int: int) -> str:
    nv = normalize_to_threshold_space(binary_str, max_int, lengthN)
    return "".join("1" if nv > t else "0" for t in thresholds)


# -------------------------
# uMUL masked stream (sequential threshold consumption)
# -------------------------

def umul_stream_sequentialA(binary_str: str,
                            thresholds_A: List[int],
                            mask_stream: str,
                            lengthN: int,
                            max_int: int) -> str:
    """
    Only consume thresholds_A when mask_stream[i] == '1'; else output '0'.
    """
    nv = normalize_to_threshold_space(binary_str, max_int, lengthN)
    out = []
    idx = 0
    for i in range(lengthN):
        if mask_stream[i] == "1":
            out.append("1" if nv > thresholds_A[idx] else "0")
            idx += 1
        else:
            out.append("0")
    return "".join(out)


# -------------------------
# LFSR baseline (auto taps + sr % N mapping)
# -------------------------

def get_lfsr_taps(bit_width: int) -> List[int]:
    primitive_polynomials = {
        4: [4, 3],
        5: [5, 3],
        6: [6, 5],
        7: [7, 6],
        8: [8, 6, 5, 4],
        9: [9, 5],
        10: [10, 7],
        11: [11, 9],
        12: [12, 11, 10, 4],
        13: [13, 12, 11, 8],
        14: [14, 13, 12, 2],
        15: [15, 14],
        16: [16, 14, 13, 11],
    }
    return primitive_polynomials.get(bit_width, [bit_width, bit_width - 1])


def lfsr_thresholds_auto(seed: int, lengthN: int) -> List[int]:
    """
      bit_width = max(4, ceil(log2(N+1)))
      taps = get_lfsr_taps(bit_width)
      update sr then output sr % N
    """
    bit_width = max(4, math.ceil(math.log2(lengthN + 1)))
    taps = get_lfsr_taps(bit_width)
    mask = (1 << bit_width) - 1

    sr = seed & mask
    if sr == 0:
        sr = 1

    out = []
    for _ in range(lengthN):
        bit = 0
        for t in taps:
            bit ^= (sr >> (t - 1)) & 1
        sr = ((sr << 1) & mask) | bit
        out.append(sr % lengthN)
    return out


def lfsr_bitstream(binary_str: str, thresholds: List[int], lengthN: int, max_int: int) -> str:
    nv = normalize_to_threshold_space(binary_str, max_int, lengthN)
    return "".join("1" if nv > t else "0" for t in thresholds)