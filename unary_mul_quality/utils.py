#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 13:23:50 2025

@author: xia
"""

# utils.py
# Common utilities for unary bitstream evaluation.

from __future__ import annotations

import random
from typing import List


def generate_random_bit() -> str:
    """Generate a random bit ('0' or '1') with equal probability."""
    return random.choice(["0", "1"])


def generate_random_8bit_nonzero() -> str:
    """Generate an 8-bit random binary string, excluding all zeros."""
    while True:
        s = "".join(generate_random_bit() for _ in range(8))
        if s != "00000000":
            return s


def binary_to_decimal(binary_str: str) -> int:
    return int(binary_str, 2)


def normalize_8bit_to_threshold_space(binary_str: str, lengthN: int) -> float:
    """
    normalized_value = (x / 255) * (lengthN - 1)
    """
    x = int(binary_str, 2)
    return (x / 255.0) * (lengthN - 1)


def print_table(title: str, table: dict, lengthN_list: List[int], methods: List[str]) -> None:
    print("\n==============================")
    print(f"=== {title} ===")
    print("==============================")

    header = "Method     | " + "  ".join([f"{N:6d}" for N in lengthN_list])
    print(header)
    print("-" * len(header))

    for m in methods:
        row_values = "  ".join([f"{v:.6f}" for v in table[m]])
        print(f"{m:<10} | {row_values}")