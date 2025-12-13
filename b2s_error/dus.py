# dus.py
# ADUS / SDUS construction and stream generation (same comparator rule).

from __future__ import annotations

import math
from math import gcd
from typing import List

from utils import normalize_to_threshold_space


def make_adus(lengthN: int) -> List[int]:
    return list(range(lengthN))


def make_sdus(lengthN: int, a: int) -> List[int]:
    assert gcd(a, lengthN) == 1
    return [((a * i) % lengthN) for i in range(lengthN)]


def stream_from_thresholds(binary_str: str, thresholds: List[int], lengthN: int, max_int: int) -> str:
    """
    Matches generate_255bit_stream_N3:
      bit[i] = 1 if normalized_value > thresholds[i] else 0
    """
    nv = normalize_to_threshold_space(binary_str, max_int, lengthN)
    return "".join("1" if nv > t else "0" for t in thresholds)


def get_default_a(lengthN: int) -> int:

    a_dict = {16: 7, 32: 19, 64: 41, 128: 83, 256: 157, 512: 323, 1024: 629}
    if lengthN not in a_dict:
        raise KeyError(f"No default a for lengthN={lengthN}")
    return a_dict[lengthN]