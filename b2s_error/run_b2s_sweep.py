#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sweep over lengthN = [16, 32, 64, 128, 256, 512, 1024]
Auto-select bit width = log2(lengthN),
Compute average B2S errors,
Plot error vs. lengthN for all methods.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from utils import bits_from_lengthN, generate_random_nbit, binary_to_decimal
from dus import generate_deterministic_uniform_sequence
from baselines import (
    binary_to_random_bitstream,
    generate_threshold_stream,
    gen_halton_thresholds_pair,
    gen_sobol_thresholds_pair,
    generate_lfsr_sequence_mbit,
    generate_bitstream_LFSR_mbit,
)
from metrics import calc_b2s_error


def simulate_b2s_avg(simulations=10000, lengthN=1024):
    BIT_WIDTH = bits_from_lengthN(lengthN)
    MAX_INT_FOR_BITS = (1 << BIT_WIDTH) - 1

    # DUS params
    a_dict = {16: 7, 32: 19, 64: 41, 128: 83, 256: 157, 512: 323, 1024: 629}
    a = a_dict[lengthN]
    ADUS = list(range(lengthN))
    proposed = generate_deterministic_uniform_sequence(lengthN, a)

    methods = ['Random', 'LFSR', 'Halton', 'Sobol', 'DUS']
    error_sum = {m: 0.0 for m in methods}

    bit_width = bits_from_lengthN(lengthN)

    for _ in range(simulations):
        # - Halton/Sobol engine created inside each simulation.
        H1, H2 = gen_halton_thresholds_pair(lengthN)
        S1, S2 = gen_sobol_thresholds_pair(lengthN)

        # Binary inputs (use BIT_WIDTH bits, excluding all-zero)
        binary_numberA = generate_random_nbit(BIT_WIDTH)
        binary_numberB = generate_random_nbit(BIT_WIDTH)
        decimal_numberA = binary_to_decimal(binary_numberA)
        decimal_numberB = binary_to_decimal(binary_numberB)
        NumberA = decimal_numberA / MAX_INT_FOR_BITS
        NumberB = decimal_numberB / MAX_INT_FOR_BITS

        # Random streams
        StreamA_random = binary_to_random_bitstream(binary_numberA, lengthN, BIT_WIDTH)
        StreamB_random = binary_to_random_bitstream(binary_numberB, lengthN, BIT_WIDTH)

        # - rA/rB generated inside each simulation.
        rA = generate_lfsr_sequence_mbit(seed=0b10100101, bit_width=bit_width, lengthN=lengthN)
        rB = generate_lfsr_sequence_mbit(seed=0b01010100, bit_width=bit_width, lengthN=lengthN)
        streamA_LFSR = generate_bitstream_LFSR_mbit(binary_numberA, rA)
        streamB_LFSR = generate_bitstream_LFSR_mbit(binary_numberB, rB)

        # Threshold streams
        streamA_H1 = generate_threshold_stream(binary_numberA, H1, lengthN, MAX_INT_FOR_BITS)
        streamB_H2 = generate_threshold_stream(binary_numberB, H2, lengthN, MAX_INT_FOR_BITS)
        streamA_S1 = generate_threshold_stream(binary_numberA, S1, lengthN, MAX_INT_FOR_BITS)
        streamB_S2 = generate_threshold_stream(binary_numberB, S2, lengthN, MAX_INT_FOR_BITS)
        streamA_ADUS = generate_threshold_stream(binary_numberA, ADUS, lengthN, MAX_INT_FOR_BITS)
        streamB_P2 = generate_threshold_stream(binary_numberB, proposed, lengthN, MAX_INT_FOR_BITS)

        e = {
            'Random': (calc_b2s_error(StreamA_random, NumberA, lengthN) + calc_b2s_error(StreamB_random, NumberB, lengthN)) / 2,
            'LFSR':   (calc_b2s_error(streamA_LFSR, NumberA, lengthN)   + calc_b2s_error(streamB_LFSR, NumberB, lengthN)) / 2,
            'Halton': (calc_b2s_error(streamA_H1, NumberA, lengthN)     + calc_b2s_error(streamB_H2, NumberB, lengthN)) / 2,
            'Sobol':  (calc_b2s_error(streamA_S1, NumberA, lengthN)     + calc_b2s_error(streamB_S2, NumberB, lengthN)) / 2,
            'DUS':    (calc_b2s_error(streamA_ADUS, NumberA, lengthN)   + calc_b2s_error(streamB_P2, NumberB, lengthN)) / 2,
        }

        for m in methods:
            error_sum[m] += e[m]

    return {m: error_sum[m] / simulations for m in methods}


def plot_b2s_errors(errors_dict, save_path="B2S_error.pdf"):
    # ================== Plot style (TCAS-I) ==================
    import matplotlib as mpl

    FONT = "Times New Roman"
    BASE_FONTSIZE = 8
    LABEL_FONTSIZE = 9
    BORDER_WIDTH = 0.8
    LINEWIDTH = 1.0
    MARKERSIZE = 5
    TICK_LENGTH = 2.5
    TICK_WIDTH = 0.7

    mpl.rcParams.update({
        "font.family": FONT,
        "font.serif": [FONT],
        "font.sans-serif": [FONT],
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,

        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    # ====== unified palette (same as MAE figure) ======
    C1 = "#22BDD2"
    C3 = "#1B78B2"
    C5 = "#9368AB"
    C7 = "#F47F1E"
    C9 = "#2DA248"

    COLOR_LFSR = C1
    COLOR_RANDOM = C3
    COLOR_HALTON = C7
    COLOR_SOBOL = C9
    COLOR_DUS = C5

    # ================== X axis: n_list (equal spacing) ==================
    n_list = np.array([4, 5, 6, 7, 8, 9, 10])

    methods = ['Random', 'LFSR', 'Halton', 'Sobol', 'DUS']

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # plot order fixed (legend order stable)
    ax.plot(n_list, errors_dict["LFSR"], 'h--',
            markerfacecolor="none", color=COLOR_LFSR,
            linewidth=LINEWIDTH, markersize=MARKERSIZE,
            label="LFSR")

    ax.plot(n_list, errors_dict["Random"], 'o--',
            markerfacecolor="none", color=COLOR_RANDOM,
            linewidth=LINEWIDTH, markersize=MARKERSIZE,
            label="Random")

    ax.plot(n_list, errors_dict["Halton"], 'd-.',
            markerfacecolor="none", color=COLOR_HALTON,
            linewidth=LINEWIDTH, markersize=MARKERSIZE,
            label="Halton")

    ax.plot(n_list, errors_dict["Sobol"], 's--',
            markerfacecolor="none", color=COLOR_SOBOL,
            linewidth=LINEWIDTH, markersize=MARKERSIZE,
            label="Sobol")

    ax.plot(n_list, errors_dict["DUS"], '^-',
            markerfacecolor="none", color=COLOR_DUS,
            linewidth=LINEWIDTH, markersize=MARKERSIZE,
            label="DUS (Proposed)")

    # ====== axis settings ======
    ax.set_xlabel("Operand Precision $n$ [bits]", fontname=FONT, fontweight="bold")
    ax.set_ylabel("B2S Error", fontname=FONT, fontweight="bold")

    ax.set_yscale("linear")

    ax.set_xticks(n_list)
    ax.set_xticklabels([str(n) for n in n_list])

    ax.set_ylim(-0.01, 0.1)

    ax.tick_params(axis='both', which='both',
                   direction='in',
                   length=TICK_LENGTH,
                   width=TICK_WIDTH)

    for spine in ax.spines.values():
        spine.set_linewidth(BORDER_WIDTH)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT)
        label.set_fontweight("bold")

    leg = ax.legend(loc="upper right", frameon=False)
    for text in leg.get_texts():
        text.set_fontname(FONT)
        text.set_fontweight("bold")

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    length_list = [16, 32, 64, 128, 256, 512, 1024]
    methods = ['Random', 'LFSR', 'Halton', 'Sobol', 'DUS']
    errors_dict = {m: [] for m in methods}

    simulations = 10000

    for lengthN in length_list:
        res = simulate_b2s_avg(simulations=simulations, lengthN=lengthN)
        for m in methods:
            errors_dict[m].append(res[m])
        print(f"lengthN={lengthN}  " + "  ".join([f"{m}={res[m]:.6f}" for m in methods]))

    plot_b2s_errors(errors_dict, save_path="B2S_error.pdf")


if __name__ == "__main__":
    main()