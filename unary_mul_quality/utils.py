#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import math
import numpy as np


def set_spine_linewidth(ax, linewidth=2):
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)


def generate_random_bit():
    """Generate a random bit, '1' or '0', with equal probability."""
    return random.choice(['0', '1'])


def generate_random_8bit():
    """Generate an 8-bit random binary number, excluding all zeros."""
    while True:
        binary_str = ''.join(generate_random_bit() for _ in range(8))
        if not all(bit == '0' for bit in binary_str):
            return binary_str


def binary_to_decimal(binary_str):
    return int(binary_str, 2)


def binary_to_random_bitstream(binary_str, lengthN):
    """Generate a random bitstream and compare with binary input."""
    random_bitstream = ''
    for _ in range(lengthN):
        random_8bit_str = generate_random_8bit()
        if int(binary_str, 2) > int(random_8bit_str, 2):
            random_bitstream += '1'
        else:
            random_bitstream += '0'
    return random_bitstream


def bits_from_lengthN(lengthN: int) -> int:
    """Return log2(lengthN), lengthN must be a power of 2."""
    if lengthN <= 1 or (lengthN & (lengthN - 1)) != 0:
        raise ValueError(f"lengthN={lengthN} is not a power of 2")
    return int(math.log2(lengthN))


def print_table(title, table, lengthN_list, methods):
    print("\n==============================")
    print(f"=== {title} ===")
    print("==============================")

    header = "Method     | " + "  ".join([f"{N:6d}" for N in lengthN_list])
    print(header)
    print("-" * len(header))

    for m in methods:
        row_values = "  ".join([f"{v:.6f}" for v in table[m]])
        print(f"{m:<10} | {row_values}")