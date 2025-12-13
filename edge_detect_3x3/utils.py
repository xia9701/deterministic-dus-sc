from __future__ import annotations
import math
import numpy as np

def unipolar_to_bipolar(p: float) -> float:
    return 2.0 * p - 1.0

def bipolar_to_unipolar(x: float) -> float:
    return (x + 1.0) / 2.0

def is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def replicate_expand(bitstream: np.ndarray, times: int) -> np.ndarray:
    return np.tile(bitstream.astype(np.uint8), max(1, times))

def rotate_expand(bitstream: np.ndarray, times: int) -> np.ndarray:
    bitstream = bitstream.astype(np.uint8)
    L = len(bitstream)
    times = max(1, times)
    extended = np.empty(L * times, dtype=np.uint8)
    for i in range(times):
        extended[i*L:(i+1)*L] = np.roll(bitstream, -i)
    return extended