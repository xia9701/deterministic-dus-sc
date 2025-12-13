# utils.py
# Utilities for B2S error sweep (power-of-two N, random n-bit input strings).

from __future__ import annotations

import math
import random


def bits_from_lengthN(lengthN: int) -> int:
    """Return BIT_WIDTH = log2(lengthN). Require lengthN is power of two."""
    if lengthN <= 1 or (lengthN & (lengthN - 1)) != 0:
        raise ValueError(f"lengthN={lengthN} is not a power of two.")
    return int(math.log2(lengthN))


def generate_random_bit() -> str:
    return random.choice(["0", "1"])


def generate_random_nbit_nonzero(bit_width: int) -> str:
    """Generate a random bit_width-bit binary string, excluding all zeros."""
    while True:
        s = "".join(generate_random_bit() for _ in range(bit_width))
        if any(ch == "1" for ch in s):
            return s


def binary_to_decimal(binary_str: str) -> int:
    return int(binary_str, 2)


def normalize_to_threshold_space(binary_str: str, max_int: int, lengthN: int) -> float:
    """
      normalized_value = int(binary_str,2) / MAX_INT_FOR_BITS * (lengthN - 1)
    """
    return (int(binary_str, 2) / float(max_int)) * (lengthN - 1)