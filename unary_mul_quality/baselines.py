#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 13:30:33 2025

@author: xia
"""

# baselines.py
# Random, LFSR, Halton, Sobol, uMUL baselines (code-faithful to your script).

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
from scipy.stats.qmc import Halton, Sobol

from utils import generate_random_8bit_nonzero, normalize_8bit_to_threshold_space


# -------------------------
# Random baseline (per-bit fresh random 8-bit threshold)
# -------------------------

def binary_to_random_bitstream(binary_str: str, lengthN: int) -> str:
    """
    EXACTLY matches your original baseline:
    for each unary bit, generate a fresh random 8-bit threshold and compare.
    """
    out = []
    x = int(binary_str, 2)
    for _ in range(lengthN):
        thr = int(generate_random_8bit_nonzero(), 2)
        out.append("1" if x > thr else "0")
    return "".join(out)


# -------------------------
# Halton / Sobol baselines (threshold arrays)
# -------------------------

def make_halton_thresholds(lengthN: int) -> Tuple[List[int], List[int]]:
    """
    Matches your per-trial behavior: instantiate engine and call random twice.
    """
    halton_engine = Halton(d=2, scramble=False)
    s1 = halton_engine.random(n=lengthN)
    s2 = halton_engine.random(n=lengthN)
    H1 = (s1[:, 0] * lengthN).astype(int).tolist()
    H2 = (s2[:, 1] * lengthN).astype(int).tolist()
    return H1, H2


def make_sobol_thresholds(lengthN: int) -> Tuple[List[int], List[int]]:
    """
    Matches your per-trial behavior: instantiate engine and call random twice.
    """
    sobol_engine = Sobol(d=2, scramble=False)
    s1 = sobol_engine.random(n=lengthN)
    s2 = sobol_engine.random(n=lengthN)
    S1 = (s1[:, 0] * lengthN).astype(int).tolist()
    S2 = (s2[:, 1] * lengthN).astype(int).tolist()
    return S1, S2


def stream_from_thresholds(binary_str: str, thresholds: List[int], lengthN: int) -> str:
    """
    Same comparator rule as your generate_255bit_stream_N3.
    """
    nv = normalize_8bit_to_threshold_space(binary_str, lengthN)
    return "".join("1" if nv > t else "0" for t in thresholds)


# -------------------------
# uMUL masked generation (sequential A consumption)
# -------------------------

def umul_masked_stream_sequentialA(binary_str: str,
                                  thresholds_A: List[int],
                                  mask_stream: str,
                                  lengthN: int) -> str:
    """
    EXACTLY matches generate_255bit_stream_N3_UGMEE_sequentialA:
    - Only consume thresholds_A when mask_stream[i] == '1'
    - Otherwise output 0
    """
    nv = normalize_8bit_to_threshold_space(binary_str, lengthN)
    out = []
    A_index = 0
    for i in range(lengthN):
        if mask_stream[i] == "1":
            out.append("1" if nv > thresholds_A[A_index] else "0")
            A_index += 1
        else:
            out.append("0")
    return "".join(out)


# -------------------------
# LFSR baseline (threshold arrays)
# -------------------------

BIT_WIDTH_LFSR = 12
LFSR_MASK = (1 << BIT_WIDTH_LFSR) - 1  

LFSR_TAPS_A = [10, 7] 
LFSR_TAPS_B = [10, 3] 


def generate_master_lfsr_sequence(taps: List[int], seed: int) -> List[int]:
    """
    Matches your original generate_master_lfsr_sequence.
    """
    sr = seed & LFSR_MASK
    if sr == 0:
        sr = 1

    seq = []
    visited = set()
    while sr not in visited:
        visited.add(sr)
        seq.append(sr)

        bit = 0
        for t in taps:
            bit ^= (sr >> (t - 1)) & 1
        sr = ((sr << 1) & LFSR_MASK) | bit

    return seq


MASTER_LFSR_A_SEQ = generate_master_lfsr_sequence(LFSR_TAPS_A, seed=0b1010010101)
MASTER_LFSR_B_SEQ = generate_master_lfsr_sequence(LFSR_TAPS_B, seed=0b1100100110)


def get_lfsr_thresholds_for_N(lengthN: int, seq: List[int]) -> List[int]:
    """
    Matches your original mapping:
      a_val = (sr * lengthN) // 2^BIT_WIDTH_LFSR
      clamp to [0, lengthN-1]
    """
    period = len(seq)
    out = []
    for i in range(lengthN):
        sr = seq[i % period]
        a_val = (sr * lengthN) // (1 << BIT_WIDTH_LFSR)
        if a_val >= lengthN:
            a_val = lengthN - 1
        out.append(a_val)
    return out


def make_lfsr_thresholds(lengthN: int) -> Tuple[List[int], List[int]]:
    A = get_lfsr_thresholds_for_N(lengthN, MASTER_LFSR_A_SEQ)
    B = get_lfsr_thresholds_for_N(lengthN, MASTER_LFSR_B_SEQ)
    return A, B


def lfsr_stream(binary_str: str, thresholds: List[int], lengthN: int) -> str:
    """
    Matches generate_255bit_stream_LFSR (same '>' comparator).
    """
    nv = normalize_8bit_to_threshold_space(binary_str, lengthN)
    return "".join("1" if nv > t else "0" for t in thresholds)