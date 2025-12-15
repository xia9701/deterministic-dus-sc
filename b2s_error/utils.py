#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import math


def bits_from_lengthN(lengthN: int) -> int:
    if lengthN <= 1 or (lengthN & (lengthN - 1)) != 0:
        raise ValueError(f"lengthN={lengthN} not right")
    return int(math.log2(lengthN))


def generate_random_bit():
    return random.choice(['0', '1'])


def generate_random_nbit(bit_width: int):
    while True:
        binary_str = ''.join(generate_random_bit() for _ in range(bit_width))
        if any(bit == '1' for bit in binary_str):
            return binary_str


def binary_to_decimal(binary_str):
    return int(binary_str, 2)