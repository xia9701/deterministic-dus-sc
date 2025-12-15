#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats.qmc import Halton, Sobol


# =========================
# Random / Threshold-based streams
# =========================
def binary_to_random_bitstream(binary_str, lengthN, bit_width: int):
    """
    Random baseline:
    For each bit, generate a fresh random n-bit threshold and compare.
    """
    from utils import generate_random_nbit
    random_bitstream = ''
    for _ in range(lengthN):
        random_nbit_str = generate_random_nbit(bit_width)
        random_bitstream += '1' if int(binary_str, 2) > int(random_nbit_str, 2) else '0'
    return random_bitstream


def generate_threshold_stream(binary_str, A, lengthN, max_int_for_bits: int):
    """
    Threshold stream:
      normalized_value = x / MAX_INT_FOR_BITS * (lengthN - 1)
      out[i] = 1 if normalized_value > A[i] else 0
    """
    normalized_value = int(binary_str, 2) / max_int_for_bits * (lengthN - 1)
    return ''.join('1' if normalized_value > num else '0' for num in A)


# =========================
# QMC sequences (Halton / Sobol)
# =========================
def gen_halton_thresholds_pair(lengthN: int):
    """
      Create a new Halton engine for EACH simulation.
      Draw twice and use H1 from [:,0] of first draw, H2 from [:,1] of second draw.
    """
    halton_engine = Halton(d=2, scramble=False)
    halton_samples_all = halton_engine.random(n=lengthN)
    halton_samples_all2 = halton_engine.random(n=lengthN)
    H1 = (halton_samples_all[:, 0] * lengthN).astype(int).tolist()
    H2 = (halton_samples_all2[:, 1] * lengthN).astype(int).tolist()
    return H1, H2


def gen_sobol_thresholds_pair(lengthN: int):
    """
      Create a new Sobol engine for EACH simulation.
      Draw twice and use S1 from [:,0] of first draw, S2 from [:,1] of second draw.
    """
    sobol_engine = Sobol(d=2, scramble=False)
    sobol_samples = sobol_engine.random(n=lengthN)
    sobol_samples2 = sobol_engine.random(n=lengthN)
    S1 = (sobol_samples[:, 0] * lengthN).astype(int).tolist()
    S2 = (sobol_samples2[:, 1] * lengthN).astype(int).tolist()
    return S1, S2


# =========================
# LFSR
# =========================
def get_lfsr_taps(bit_width):
    primitive_polynomials = {
        4: [4, 3],
        5: [5, 3],
        6: [6, 5],
        7: [7, 6],
        8: [8, 6, 5, 4],
        9: [9, 5],
        10: [10, 7],
        11: [11, 9],
        12: [12, 11, 10, 4],
        13: [13, 12, 11, 8],
        14: [14, 13, 12, 2],
        15: [15, 14],
        16: [16, 14, 13, 11],
    }
    return primitive_polynomials.get(bit_width, [bit_width, bit_width - 1])


def generate_lfsr_sequence_mbit(seed, bit_width, lengthN):

    taps = get_lfsr_taps(bit_width)
    sr = seed & ((1 << bit_width) - 1)
    if sr == 0:
        sr = 1

    seq = []
    for _ in range(lengthN):
        fb = 0
        for t in taps:
            fb ^= (sr >> (t - 1)) & 1
        sr = ((sr << 1) & ((1 << bit_width) - 1)) | fb
        seq.append(sr)
    return seq


def generate_bitstream_LFSR_mbit(binary_str, r_seq):
    """
    LFSR bitstream:
      out[i] = 1 if r_seq[i] < x_int else 0
    """
    x_int = int(binary_str, 2)
    return ''.join('1' if r < x_int else '0' for r in r_seq)