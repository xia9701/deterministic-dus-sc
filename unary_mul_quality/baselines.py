#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats.qmc import Halton, Sobol


def generate_255bit_stream_N3(binary_str, A, lengthN):
    """Generate a bitstream by thresholding sequence A (0..lengthN-1)."""
    normalized_value = int(binary_str, 2) / 255 * (lengthN - 1)
    random_bitstream = ''
    for num in A:
        if normalized_value > num:
            random_bitstream += '1'
        else:
            random_bitstream += '0'
    return random_bitstream


# ============================================================
# LFSR baseline (m-bit LFSR, r_t in [1, 2^m-1], compare r < x_m)
# ============================================================
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
    """
    Generate m-bit LFSR random sequence r_t ∈ [1, 2^m - 1]
    """
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


def generate_bitstream_LFSR_mbit_from8(binary_str_8bit, r_seq, bit_width):
    """
    LFSR bitstream generation (r < x),
    x is 8-bit (0..255), mapped to m-bit domain first.
    """
    x8 = int(binary_str_8bit, 2)  # 0..255
    x_m = int(round(x8 / 255.0 * ((1 << bit_width) - 1)))  # 0..(2^m-1)
    return ''.join('1' if r < x_m else '0' for r in r_seq)


# ============================================================
# QMC thresholds (Halton/Sobol) — keep EXACT same calling style
# ============================================================
def gen_halton_thresholds_pair(lengthN):
    """
    Match original script behavior:
      halton_engine = Halton(d=2, scramble=False)
      halton_samples_all  = engine.random(n=lengthN)
      halton_samples_all2 = engine.random(n=lengthN)
      H1 from [:,0], H2 from [:,1] of the SECOND draw
    """
    halton_engine = Halton(d=2, scramble=False)
    halton_samples_all = halton_engine.random(n=lengthN)
    halton_samples_all2 = halton_engine.random(n=lengthN)
    H1 = (halton_samples_all[:, 0] * lengthN).astype(int).tolist()
    H2 = (halton_samples_all2[:, 1] * lengthN).astype(int).tolist()
    return H1, H2


def gen_sobol_thresholds_pair(lengthN):
    """
    Match original script behavior:
      sobol_engine = Sobol(d=2, scramble=False)
      sobol_samples  = engine.random(n=lengthN)
      sobol_samples2 = engine.random(n=lengthN)
      S1 from [:,0], S2 from [:,1] of the SECOND draw
    """
    sobol_engine = Sobol(d=2, scramble=False)
    sobol_samples = sobol_engine.random(n=lengthN)
    sobol_samples2 = sobol_engine.random(n=lengthN)
    S1 = (sobol_samples[:, 0] * lengthN).astype(int).tolist()
    S2 = (sobol_samples2[:, 1] * lengthN).astype(int).tolist()
    return S1, S2