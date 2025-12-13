#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 13:29:27 2025

@author: xia
"""

# dus.py
# ADUS/SDUS construction + stream generation from threshold arrays.

from __future__ import annotations

from math import gcd
from typing import List

from utils import normalize_8bit_to_threshold_space


def make_adus(lengthN: int) -> List[int]:
    """ADUS = [0, 1, ..., N-1]."""
    return list(range(lengthN))


def make_sdus(lengthN: int, a: int) -> List[int]:
    """
    SDUS = [(a*i) % N], gcd(a, N) == 1
    (Matches your generate_deterministic_uniform_sequence.)
    """
    assert gcd(a, lengthN) == 1
    return [((a * i) % lengthN) for i in range(lengthN)]


def stream_from_thresholds(binary_str: str, thresholds: List[int], lengthN: int) -> str:
    """
    Implements:
      normalized_value = (x/255)*(N-1)
      bit[i] = 1 if normalized_value > thresholds[i] else 0
    (Matches generate_255bit_stream_N3.)
    """
    normalized_value = normalize_8bit_to_threshold_space(binary_str, lengthN)
    # thresholds length is assumed == lengthN
    return "".join("1" if normalized_value > t else "0" for t in thresholds)