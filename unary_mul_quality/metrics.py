#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random


def compute_scc(X, Y):
    assert len(X) == len(Y)
    L = len(X)
    a = sum(1 for x, y in zip(X, Y) if x == '1' and y == '1')
    b = sum(1 for x, y in zip(X, Y) if x == '1' and y == '0')
    c = sum(1 for x, y in zip(X, Y) if x == '0' and y == '1')
    d = sum(1 for x, y in zip(X, Y) if x == '0' and y == '0')

    numerator = a * d - b * c
    if a * d > b * c:
        denominator = L * min(a + b, a + c) - (a + b) * (a + c)
    else:
        denominator = (a + b) * (a + c) - L * max(a - d, 0)

    if denominator == 0:
        return 0.0
    return abs(numerator / denominator)


def compute_zce(X, Y):
    assert len(X) == len(Y)
    L = len(X)

    a = sum(1 for x, y in zip(X, Y) if x == '1' and y == '1')
    b = sum(1 for x, y in zip(X, Y) if x == '1' and y == '0')
    c = sum(1 for x, y in zip(X, Y) if x == '0' and y == '1')

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


def F_mul(S1, S2):
    """Perform bitwise AND on two bitstreams."""
    return ''.join('1' if S1[i] == '1' and S2[i] == '1' else '0' for i in range(len(S1)))


def compute_stream_mul_error(streamA, streamB, ref_val, lengthN):
    """
      y_hat = ones/(lengthN-1)
    """
    result = F_mul(streamA, streamB)
    error = abs(ref_val - result.count('1') / (lengthN - 1))
    return error


# =========================
# Scaled add (MUX) + MAE
# =========================
def F_add_mux(SA, SB, Ssel):
    """Scaled add via MUX: if Ssel[i]=='1' pick SA[i], else pick SB[i]."""
    assert len(SA) == len(SB) == len(Ssel)
    return ''.join(SA[i] if Ssel[i] == '1' else SB[i] for i in range(len(SA)))


def compute_stream_add_error_scaled(streamA, streamB, sel_stream, ref_add, lengthN):
    """
    Scaled add MAE:
      out = MUX(sel, A, B)
      y_hat = ones(out)/(lengthN-1)   # keep denominator consistent with MUL
      err = |ref_add - y_hat|
    """
    out = F_add_mux(streamA, streamB, sel_stream)
    y_hat = out.count('1') / (lengthN - 1)
    return abs(ref_add - y_hat)


def fresh_unbiased_sel_stream(lengthN):
    """Fresh unbiased MUX select stream per simulation."""
    return ''.join(random.choice(['0', '1']) for _ in range(lengthN))