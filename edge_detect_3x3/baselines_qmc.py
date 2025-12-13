from __future__ import annotations
import math
import numpy as np
from scipy.stats.qmc import Halton, Sobol

# SDUS parameter table
A_DICT = {16:7, 32:19, 64:41, 128:83, 256:157, 512:323, 1024:629}

def generate_sdus(N: int, a: int) -> np.ndarray:
    from math import gcd
    assert gcd(a, N) == 1
    return np.array([(a * i) % N for i in range(N)], dtype=np.int32)

def random8_thresholds(lengthN: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rnd8 = np.random.randint(0, 256, size=lengthN, dtype=np.int32)
    else:
        rnd8 = rng.integers(0, 256, size=lengthN, dtype=np.int32)
    A = (rnd8 * (lengthN - 1) // 255).astype(np.int32)
    return A

def random8_thresholds_pair(lengthN: int, rng: np.random.Generator | None = None):
    return random8_thresholds(lengthN, rng), random8_thresholds(lengthN, rng)

def sobol_thresholds_pair(lengthN: int, *, seed=None) -> tuple[np.ndarray, np.ndarray]:
    eng = Sobol(d=2, scramble=False, seed=seed)
    S = eng.random(n=lengthN)
    A1 = np.clip((S[:, 0] * lengthN).astype(int), 0, lengthN - 1).astype(np.int32)
    A2 = np.clip((S[:, 1] * lengthN).astype(int), 0, lengthN - 1).astype(np.int32)
    return A1, A2

def halton_thresholds_pair(lengthN: int, *, seed=None) -> tuple[np.ndarray, np.ndarray]:
    eng = Halton(d=2, scramble=False, seed=seed)
    H = eng.random(n=lengthN)
    A1 = np.clip((H[:, 0] * lengthN).astype(int), 0, lengthN - 1).astype(np.int32)
    A2 = np.clip((H[:, 1] * lengthN).astype(int), 0, lengthN - 1).astype(np.int32)
    return A1, A2

# ---- LFSR taps ----
def get_lfsr_taps(bit_width: int):
    primitive_polynomials = {
        4:[4,3], 5:[5,3], 6:[6,5], 7:[7,6], 8:[8,6,5,4], 9:[9,5], 10:[10,7], 11:[11,9],
        12:[12,11,10,4], 13:[13,12,11,8], 14:[14,13,12,2], 15:[15,14], 16:[16,14,13,11]
    }
    if bit_width not in primitive_polynomials:
        raise ValueError(f"No taps defined for bit width {bit_width}")
    return primitive_polynomials[bit_width]

def _lfsr_step(sr: int, taps: list[int], mask: int) -> int:
    fb = 0
    for t in taps:
        fb ^= (sr >> (t - 1)) & 1
    sr = ((sr << 1) & mask) | fb
    return sr if sr != 0 else 1

def generate_lfsr_sequence_auto(seed: int, lengthN: int) -> np.ndarray:
    """
    - Use 16-bit LFSR (taps [16,14,13,11]) to produce sr values.
    - Map to thresholds by high-bit scaling: A = (vals * N) >> 16.
    """
    bit_width = 16
    mask = (1 << bit_width) - 1
    taps = get_lfsr_taps(bit_width)
    sr = seed & mask
    if sr == 0:
        sr = 1

    vals = np.empty(lengthN, dtype=np.uint32)
    for i in range(lengthN):
        sr = _lfsr_step(sr, taps, mask)
        vals[i] = sr

    A = ((vals.astype(np.uint32) * lengthN) >> bit_width).astype(np.int32)
    return A