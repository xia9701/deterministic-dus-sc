# run_b2s_sweep.py
# Sweep N and plot mean B2S error for different methods.

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from utils import bits_from_lengthN, generate_random_nbit_nonzero, binary_to_decimal
from metrics import b2s_error_from_stream
from dus import make_adus, make_sdus, stream_from_thresholds as dus_stream, get_default_a
from baselines import (
    random_bitstream,
    halton_thresholds,
    sobol_thresholds,
    stream_from_thresholds,
    lfsr_thresholds_auto,
    lfsr_bitstream,
    umul_stream_sequentialA,
)


def simulate_b2s_avg(simulations: int, lengthN: int) -> Dict[str, float]:
    """
    - BIT_WIDTH = log2(N)
    - MAX_INT_FOR_BITS = 2^BIT_WIDTH - 1
    - Per trial: new Halton/Sobol engines and thresholds
    - Random baseline: per-bit fresh random n-bit threshold
    - LFSR: auto taps + sr%N mapping
    - uMUL: mask stream = A_S1, thresholds = S2
    - Error: mean of A/B errors, each uses ones/(N-1)
    """
    bit_width = bits_from_lengthN(lengthN)
    max_int = (1 << bit_width) - 1

    a = get_default_a(lengthN)
    adus = make_adus(lengthN)
    sdus = make_sdus(lengthN, a)

    methods = ["Random", "LFSR", "Halton", "Sobol", "uMUL", "DUS"]
    error_sum = {m: 0.0 for m in methods}

    for _ in range(simulations):
        # Threshold arrays (per trial)
        H1, H2 = halton_thresholds(lengthN)
        S1, S2 = sobol_thresholds(lengthN)

        # Random n-bit inputs
        bA = generate_random_nbit_nonzero(bit_width)
        bB = generate_random_nbit_nonzero(bit_width)
        A = binary_to_decimal(bA) / float(max_int)
        B = binary_to_decimal(bB) / float(max_int)

        # Streams
        A_rand = random_bitstream(bA, lengthN, bit_width)
        B_rand = random_bitstream(bB, lengthN, bit_width)

        lfsr_A = lfsr_thresholds_auto(seed=0b10100101, lengthN=lengthN)
        lfsr_B = lfsr_thresholds_auto(seed=0b01010100, lengthN=lengthN)
        A_lfsr = lfsr_bitstream(bA, lfsr_A, lengthN, max_int)
        B_lfsr = lfsr_bitstream(bB, lfsr_B, lengthN, max_int)

        A_hal = stream_from_thresholds(bA, H1, lengthN, max_int)
        B_hal = stream_from_thresholds(bB, H2, lengthN, max_int)

        A_sob = stream_from_thresholds(bA, S1, lengthN, max_int)
        B_sob = stream_from_thresholds(bB, S2, lengthN, max_int)

        A_adus = dus_stream(bA, adus, lengthN, max_int)
        B_sdus = dus_stream(bB, sdus, lengthN, max_int)

        B_umul = umul_stream_sequentialA(bB, S2, A_sob, lengthN, max_int)

        # B2S error (mean of A and B)
        e = {
            "Random": (b2s_error_from_stream(A_rand, A, lengthN) + b2s_error_from_stream(B_rand, B, lengthN)) / 2,
            "LFSR":   (b2s_error_from_stream(A_lfsr, A, lengthN) + b2s_error_from_stream(B_lfsr, B, lengthN)) / 2,
            "Halton": (b2s_error_from_stream(A_hal, A, lengthN) + b2s_error_from_stream(B_hal, B, lengthN)) / 2,
            "Sobol":  (b2s_error_from_stream(A_sob, A, lengthN) + b2s_error_from_stream(B_sob, B, lengthN)) / 2,
            "uMUL":   (b2s_error_from_stream(A_sob, A, lengthN) + b2s_error_from_stream(B_umul, B, lengthN)) / 2,
            "DUS":    (b2s_error_from_stream(A_adus, A, lengthN) + b2s_error_from_stream(B_sdus, B, lengthN)) / 2,
        }

        for m in e:
            error_sum[m] += e[m]

    return {m: error_sum[m] / simulations for m in error_sum}


def main() -> None:
    # Optional: seeds improve reproducibility for Random baseline.
    random.seed(0)
    np.random.seed(0)

    length_list = [16, 32, 64, 128, 256, 512, 1024]
    methods = ["Random", "LFSR", "Halton", "Sobol", "uMUL", "DUS"]
    errors = {m: [] for m in methods}

    simulations = 1000  # adjust as needed

    for N in length_list:
        res = simulate_b2s_avg(simulations=simulations, lengthN=N)
        for m in methods:
            errors[m].append(res[m])
        print(f"lengthN={N}  " + "  ".join([f"{m}={res[m]:.6f}" for m in methods]))

    # Plot
    plt.figure(figsize=(8, 5))
    markers = ["o", "x", "^", "s", "D", "v"]
    for m, mk in zip(methods, markers):
        plt.plot(length_list, errors[m], marker=mk, label=m)

    plt.xscale("log", base=2)
    plt.xlabel("lengthN (bitstream length)")
    plt.ylabel("B2S Error (mean of A & B)")
    plt.title("B2S Error vs Bitstream Length")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()