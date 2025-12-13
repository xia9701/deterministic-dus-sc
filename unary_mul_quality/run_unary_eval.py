#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 13:31:43 2025

@author: xia
"""

# run_unary_eval.py
# Entry point: sweep N and print MAE/SCC/ZCE tables.

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from utils import generate_random_8bit_nonzero, binary_to_decimal, print_table
from dus import make_adus, make_sdus, stream_from_thresholds
from baselines import (
    binary_to_random_bitstream,
    make_lfsr_thresholds,
    lfsr_stream,
    make_halton_thresholds,
    make_sobol_thresholds,
    stream_from_thresholds as stream_from_thresholds_baseline,
    umul_masked_stream_sequentialA,
)
from metrics import compute_scc, compute_zce, compute_stream_mul_error


def simulate_and_collect(simulations: int,
                         lengthN: int,
                         adus: List[int],
                         sdus: List[int]) -> Dict:
    # LFSR thresholds (fixed per N)
    lfsr_A_thr, lfsr_B_thr = make_lfsr_thresholds(lengthN)

    # records
    mul_err = {k: [] for k in ["Random", "LFSR", "Halton", "Sobol", "uMUL", "DUS"]}
    scc = {k: [] for k in ["Random", "LFSR", "Halton", "Sobol", "uMUL", "DUS"]}
    zce = {k: [] for k in ["Random", "LFSR", "Halton", "Sobol", "uMUL", "DUS"]}

    for _ in range(simulations):

        H1, H2 = make_halton_thresholds(lengthN)
        S1, S2 = make_sobol_thresholds(lengthN)

        # Random 8-bit inputs (non-zero)
        bA = generate_random_8bit_nonzero()
        bB = generate_random_8bit_nonzero()
        A = binary_to_decimal(bA) / 255.0
        B = binary_to_decimal(bB) / 255.0
        ref_val = A * B

        # Streams for A
        A_rand = binary_to_random_bitstream(bA, lengthN)
        A_lfsr = lfsr_stream(bA, lfsr_A_thr, lengthN)
        A_hal = stream_from_thresholds_baseline(bA, H1, lengthN)
        A_sob = stream_from_thresholds_baseline(bA, S1, lengthN)
        A_adus = stream_from_thresholds(bA, adus, lengthN)

        # Streams for B
        B_rand = binary_to_random_bitstream(bB, lengthN)
        B_lfsr = lfsr_stream(bB, lfsr_B_thr, lengthN)
        B_hal = stream_from_thresholds_baseline(bB, H2, lengthN)
        B_sob = stream_from_thresholds_baseline(bB, S2, lengthN)
        B_sdus = stream_from_thresholds(bB, sdus, lengthN)

        # uMUL masked: use S1 as mask stream and consume S2 thresholds sequentially
        # (Matches: generate_..._UGMEE_sequentialA(binary_numberB, S2, streamA_S1))
        B_umul = umul_masked_stream_sequentialA(bB, S2, A_sob, lengthN)

        # MUL MAE (AND)
        mul_err["Random"].append(compute_stream_mul_error(A_rand, B_rand, ref_val, lengthN))
        mul_err["LFSR"].append(compute_stream_mul_error(A_lfsr, B_lfsr, ref_val, lengthN))
        mul_err["Halton"].append(compute_stream_mul_error(A_hal, B_hal, ref_val, lengthN))
        mul_err["Sobol"].append(compute_stream_mul_error(A_sob, B_sob, ref_val, lengthN))
        mul_err["uMUL"].append(compute_stream_mul_error(A_sob, B_umul, ref_val, lengthN))
        mul_err["DUS"].append(compute_stream_mul_error(A_adus, B_sdus, ref_val, lengthN))

        # SCC / ZCE
        scc["Random"].append(compute_scc(A_rand, B_rand))
        scc["LFSR"].append(compute_scc(A_lfsr, B_lfsr))
        scc["Halton"].append(compute_scc(A_hal, B_hal))
        scc["Sobol"].append(compute_scc(A_sob, B_sob))
        scc["uMUL"].append(compute_scc(A_sob, B_umul))
        scc["DUS"].append(compute_scc(A_adus, B_sdus))

        zce["Random"].append(compute_zce(A_rand, B_rand))
        zce["LFSR"].append(compute_zce(A_lfsr, B_lfsr))
        zce["Halton"].append(compute_zce(A_hal, B_hal))
        zce["Sobol"].append(compute_zce(A_sob, B_sob))
        zce["uMUL"].append(compute_zce(A_sob, B_umul))
        zce["DUS"].append(compute_zce(A_adus, B_sdus))

    return {"MUL": mul_err, "SCC": scc, "ZCE": zce}


def main() -> None:

    random.seed(0)
    np.random.seed(0)

    simulations = 1000
    lengthN_list = [16, 32, 64, 128, 256, 512, 1024]
    methods = ["Random", "LFSR", "Halton", "Sobol", "uMUL", "DUS"]


    a_dict = {16: 7, 32: 19, 64: 41, 128: 83, 256: 157, 512: 323, 1024: 629}

    table_MAE = {m: [] for m in methods}
    table_SCC = {m: [] for m in methods}
    table_ZCE = {m: [] for m in methods}

    for lengthN in lengthN_list:
        print(f"\n[ Running N = {lengthN} ]")

        a = a_dict[lengthN]
        adus = make_adus(lengthN)
        sdus = make_sdus(lengthN, a)

        results = simulate_and_collect(simulations, lengthN, adus, sdus)

        for m in methods:
            table_MAE[m].append(float(np.mean(results["MUL"][m])))
            table_SCC[m].append(float(np.mean(results["SCC"][m])))
            table_ZCE[m].append(float(np.mean(results["ZCE"][m])))

    print_table("MUL MAE Table", table_MAE, lengthN_list, methods)
    print_table("SCC Table", table_SCC, lengthN_list, methods)
    print_table("ZCE Table", table_ZCE, lengthN_list, methods)


if __name__ == "__main__":
    main()