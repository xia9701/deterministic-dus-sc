#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unary stream quality evaluation:
- MUL MAE (AND)
- ADD MAE (scaled add via MUX)
- SCC
- ZCE
"""

import random
import numpy as np

from utils import generate_random_8bit, binary_to_decimal, binary_to_random_bitstream, bits_from_lengthN, print_table
from baselines import (
    gen_halton_thresholds_pair, gen_sobol_thresholds_pair,
    generate_255bit_stream_N3,
    generate_lfsr_sequence_mbit, generate_bitstream_LFSR_mbit_from8
)
from dus import generate_deterministic_uniform_sequence
from metrics import (
    compute_scc, compute_zce,
    compute_stream_mul_error,
    compute_stream_add_error_scaled,
    fresh_unbiased_sel_stream
)


def simulate_and_collect_errors(simulations=10, lengthN=1024, *, ADUS=None, proposed=None):
    """Run simulations and collect MUL errors + scaled-add MAE + SCC + ZCE."""
    if ADUS is None or proposed is None:
        raise ValueError("ADUS and proposed DUS thresholds must be provided")

    NumberAA = []
    NumberBB = []
    ref_number = []

    # MUL errors
    error_MUL_LFSR1_LFSR2 = []
    error_MUL_H1_H2 = []
    error_MUL_S1_S2 = []
    error_MUL_N2_S2 = []
    error_MUL_Random = []

    # ADD (scaled) errors
    error_ADD_LFSR1_LFSR2 = []
    error_ADD_H1_H2 = []
    error_ADD_S1_S2 = []
    error_ADD_N2_S2 = []
    error_ADD_Random = []

    # SCC
    scc_Random_list = []
    scc_LFSR_list = []
    scc_Halton_list = []
    scc_Sobol_list = []
    scc_DUS_list = []

    # ZCE
    zce_Random_list = []
    zce_LFSR_list = []
    zce_Halton_list = []
    zce_Sobol_list = []
    zce_DUS_list = []

    # LFSR r-sequences
    bit_width = bits_from_lengthN(lengthN)
    rA = generate_lfsr_sequence_mbit(seed=0b10100101, bit_width=bit_width, lengthN=lengthN)
    rB = generate_lfsr_sequence_mbit(seed=0b01010100, bit_width=bit_width, lengthN=lengthN)

    for _ in range(simulations):
        # Halton thresholds
        H1, H2 = gen_halton_thresholds_pair(lengthN)

        # Sobol thresholds
        S1, S2 = gen_sobol_thresholds_pair(lengthN)

        # Binary inputs
        binary_numberA = generate_random_8bit()
        binary_numberB = generate_random_8bit()
        decimal_numberA = binary_to_decimal(binary_numberA)
        decimal_numberB = binary_to_decimal(binary_numberB)
        NumberA = decimal_numberA / 255
        NumberB = decimal_numberB / 255
        NumberAA.append(NumberA)
        NumberBB.append(NumberB)

        # A streams
        StreamA_random = binary_to_random_bitstream(binary_numberA, lengthN)
        streamA_LFSR = generate_bitstream_LFSR_mbit_from8(binary_numberA, rA, bit_width)
        streamA_H1 = generate_255bit_stream_N3(binary_numberA, H1, lengthN)
        streamA_S1 = generate_255bit_stream_N3(binary_numberA, S1, lengthN)
        streamA_ADUS = generate_255bit_stream_N3(binary_numberA, ADUS, lengthN)

        # B streams
        StreamB_random = binary_to_random_bitstream(binary_numberB, lengthN)
        streamB_LFSR = generate_bitstream_LFSR_mbit_from8(binary_numberB, rB, bit_width)
        streamB_H2 = generate_255bit_stream_N3(binary_numberB, H2, lengthN)
        streamB_S2 = generate_255bit_stream_N3(binary_numberB, S2, lengthN)
        streamB_P2 = generate_255bit_stream_N3(binary_numberB, proposed, lengthN)

        # References
        ref_mul = NumberA * NumberB
        ref_add = 0.5 * (NumberA + NumberB)  # scaled add reference
        ref_number.append(ref_mul)

        # MUL Errors
        error_MUL_Random.append(compute_stream_mul_error(StreamA_random, StreamB_random, ref_mul, lengthN))
        error_MUL_LFSR1_LFSR2.append(compute_stream_mul_error(streamA_LFSR, streamB_LFSR, ref_mul, lengthN))
        error_MUL_H1_H2.append(compute_stream_mul_error(streamA_H1, streamB_H2, ref_mul, lengthN))
        error_MUL_S1_S2.append(compute_stream_mul_error(streamA_S1, streamB_S2, ref_mul, lengthN))
        error_MUL_N2_S2.append(compute_stream_mul_error(streamA_ADUS, streamB_P2, ref_mul, lengthN))

        # ADD (Scaled) Errors
        sel_stream = fresh_unbiased_sel_stream(lengthN)
        error_ADD_Random.append(compute_stream_add_error_scaled(StreamA_random, StreamB_random, sel_stream, ref_add, lengthN))
        error_ADD_LFSR1_LFSR2.append(compute_stream_add_error_scaled(streamA_LFSR, streamB_LFSR, sel_stream, ref_add, lengthN))
        error_ADD_H1_H2.append(compute_stream_add_error_scaled(streamA_H1, streamB_H2, sel_stream, ref_add, lengthN))
        error_ADD_S1_S2.append(compute_stream_add_error_scaled(streamA_S1, streamB_S2, sel_stream, ref_add, lengthN))
        error_ADD_N2_S2.append(compute_stream_add_error_scaled(streamA_ADUS, streamB_P2, sel_stream, ref_add, lengthN))

        # SCC
        scc_Random_list.append(compute_scc(StreamA_random, StreamB_random))
        scc_LFSR_list.append(compute_scc(streamA_LFSR, streamB_LFSR))
        scc_Halton_list.append(compute_scc(streamA_H1, streamB_H2))
        scc_Sobol_list.append(compute_scc(streamA_S1, streamB_S2))
        scc_DUS_list.append(compute_scc(streamA_ADUS, streamB_P2))

        # ZCE
        zce_Random_list.append(compute_zce(StreamA_random, StreamB_random))
        zce_LFSR_list.append(compute_zce(streamA_LFSR, streamB_LFSR))
        zce_Halton_list.append(compute_zce(streamA_H1, streamB_H2))
        zce_Sobol_list.append(compute_zce(streamA_S1, streamB_S2))
        zce_DUS_list.append(compute_zce(streamA_ADUS, streamB_P2))

    return {
        'NumberAA': NumberAA,
        'NumberBB': NumberBB,
        'ref_number': ref_number,
        'MUL_errors': {
            'Random': error_MUL_Random,
            'LFSR': error_MUL_LFSR1_LFSR2,
            'Halton': error_MUL_H1_H2,
            'Sobol': error_MUL_S1_S2,
            'DUS': error_MUL_N2_S2
        },
        'ADD_errors_scaled': {  # NEW
            'Random': error_ADD_Random,
            'LFSR': error_ADD_LFSR1_LFSR2,
            'Halton': error_ADD_H1_H2,
            'Sobol': error_ADD_S1_S2,
            'DUS': error_ADD_N2_S2
        },
        'SCC': {
            'Random': scc_Random_list,
            'LFSR': scc_LFSR_list,
            'Halton': scc_Halton_list,
            'Sobol': scc_Sobol_list,
            'DUS': scc_DUS_list,
        },
        'ZCE': {
            'Random': zce_Random_list,
            'LFSR': zce_LFSR_list,
            'Halton': zce_Halton_list,
            'Sobol': zce_Sobol_list,
            'DUS': zce_DUS_list,
        }
    }


def main():
    # =============================================
    # Sweep bitstream length
    # =============================================
    simulations = 10000
    lengthN_list = [16, 32, 64, 128, 256, 512, 1024]
    methods = ["Random", "LFSR", "Halton", "Sobol", "DUS"]

    table_MAE = {m: [] for m in methods}
    table_ADD_MAE = {m: [] for m in methods}  # NEW
    table_SCC = {m: [] for m in methods}
    table_ZCE = {m: [] for m in methods}

    for lengthN in lengthN_list:
        print(f"\n[ Running N = {lengthN} ]")

        # DUS parameter
        a_dict = {16: 7, 32: 19, 64: 41, 128: 83, 256: 157, 512: 323, 1024: 629}
        a = a_dict[lengthN]

        ADUS = list(range(lengthN))
        proposed = generate_deterministic_uniform_sequence(lengthN, a)

        results = simulate_and_collect_errors(simulations=simulations, lengthN=lengthN, ADUS=ADUS, proposed=proposed)

        # MUL MAE
        table_MAE["Random"].append(np.mean(results["MUL_errors"]["Random"]))
        table_MAE["LFSR"].append(np.mean(results["MUL_errors"]["LFSR"]))
        table_MAE["Halton"].append(np.mean(results["MUL_errors"]["Halton"]))
        table_MAE["Sobol"].append(np.mean(results["MUL_errors"]["Sobol"]))
        table_MAE["DUS"].append(np.mean(results["MUL_errors"]["DUS"]))

        # ADD (Scaled) MAE
        table_ADD_MAE["Random"].append(np.mean(results["ADD_errors_scaled"]["Random"]))
        table_ADD_MAE["LFSR"].append(np.mean(results["ADD_errors_scaled"]["LFSR"]))
        table_ADD_MAE["Halton"].append(np.mean(results["ADD_errors_scaled"]["Halton"]))
        table_ADD_MAE["Sobol"].append(np.mean(results["ADD_errors_scaled"]["Sobol"]))
        table_ADD_MAE["DUS"].append(np.mean(results["ADD_errors_scaled"]["DUS"]))

        # SCC
        table_SCC["Random"].append(np.mean(results["SCC"]["Random"]))
        table_SCC["LFSR"].append(np.mean(results["SCC"]["LFSR"]))
        table_SCC["Halton"].append(np.mean(results["SCC"]["Halton"]))
        table_SCC["Sobol"].append(np.mean(results["SCC"]["Sobol"]))
        table_SCC["DUS"].append(np.mean(results["SCC"]["DUS"]))

        # ZCE
        table_ZCE["Random"].append(np.mean(results["ZCE"]["Random"]))
        table_ZCE["LFSR"].append(np.mean(results["ZCE"]["LFSR"]))
        table_ZCE["Halton"].append(np.mean(results["ZCE"]["Halton"]))
        table_ZCE["Sobol"].append(np.mean(results["ZCE"]["Sobol"]))
        table_ZCE["DUS"].append(np.mean(results["ZCE"]["DUS"]))

    print_table("MUL MAE Table", table_MAE, lengthN_list, methods)
    print_table("ADD (Scaled, MUX) MAE Table", table_ADD_MAE, lengthN_list, methods)
    print_table("SCC Table", table_SCC, lengthN_list, methods)
    print_table("ZCE Table", table_ZCE, lengthN_list, methods)


if __name__ == "__main__":
    main()