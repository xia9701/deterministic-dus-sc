#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 13:24:25 2025

@author: xia
"""

# metrics.py
# Metrics: SCC, ZCE, unary multiplication MAE (AND), etc.

from __future__ import annotations

from typing import Tuple


def _count_abcd(X: str, Y: str) -> Tuple[int, int, int, int]:
    assert len(X) == len(Y)
    a = sum(1 for x, y in zip(X, Y) if x == "1" and y == "1")
    b = sum(1 for x, y in zip(X, Y) if x == "1" and y == "0")
    c = sum(1 for x, y in zip(X, Y) if x == "0" and y == "1")
    d = sum(1 for x, y in zip(X, Y) if x == "0" and y == "0")
    return a, b, c, d


def compute_scc(X: str, Y: str) -> float:

    L = len(X)
    a, b, c, d = _count_abcd(X, Y)

    numerator = a * d - b * c
    if a * d > b * c:
        denominator = L * min(a + b, a + c) - (a + b) * (a + c)
    else:
        denominator = (a + b) * (a + c) - L * max(a - d, 0)

    if denominator == 0:
        return 0.0
    return abs(numerator / denominator)


def compute_zce(X: str, Y: str) -> float:

    L = len(X)
    a, b, c, _d = _count_abcd(X, Y)

    P_X = (a + b) / L
    P_Y = (a + c) / L
    P_XY = a / L
    delta = P_XY - (P_X * P_Y)

    raw_product = P_X * P_Y
    quantized_product = round(raw_product * L) / L
    delta_0 = quantized_product - raw_product

    if delta == 0:
        return 0.0

    zce = abs(delta * (1 - abs(delta_0 / delta)))
    return zce


def bitwise_and_mul(S1: str, S2: str) -> str:
    """Unary multiplication by bitwise AND (unipolar, uncorrelated assumption)."""
    assert len(S1) == len(S2)
    return "".join("1" if (S1[i] == "1" and S2[i] == "1") else "0" for i in range(len(S1)))


def compute_stream_mul_error(streamA: str, streamB: str, ref_val: float, lengthN: int) -> float:
    """
    error = |ref - (#ones / (lengthN - 1))|
    """
    result = bitwise_and_mul(streamA, streamB)
    return abs(ref_val - (result.count("1") / (lengthN - 1)))