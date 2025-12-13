from __future__ import annotations
import math
import numpy as np

from utils import is_pow2
from baselines_qmc import (
    A_DICT, generate_sdus,
    sobol_thresholds_pair, halton_thresholds_pair,
    random8_thresholds_pair, random8_thresholds,
    generate_lfsr_sequence_auto
)
from htc import htc_build_fsm_schedule_sobol, htc_encode_TB, htc_encode_RB_by_FSM
from vlsi22 import vlsi22_generate_stream_for_value

class BitstreamGenerator:
    """
    Single-lane generator: p in [0,1] -> {0,1}^N
    Integer-domain compare, plus p=0/1 saturation.
    """
    def __init__(self, method: str, lengthN: int, *, sdus_a: int | None = None,
                 lfsr_seed: int = 0xA5, A_override: np.ndarray | None = None):
        self.method = method.lower()
        self.lengthN = lengthN

        if A_override is not None:
            self.A = np.asarray(A_override, dtype=np.int32)
            return

        if self.method == "adus":
            self.A = np.arange(lengthN, dtype=np.int32)
        elif self.method == "sdus":
            if sdus_a is None:
                sdus_a = A_DICT.get(lengthN)
            if sdus_a is None:
                raise ValueError(f"No SDUS a for N={lengthN}")
            self.A = generate_sdus(lengthN, sdus_a)
        elif self.method == "random8":
            self.A = random8_thresholds(lengthN)
        elif self.method == "lfsr":
            self.A = generate_lfsr_sequence_auto(seed=lfsr_seed, lengthN=lengthN)
        elif self.method == "halton":
            A1, _ = halton_thresholds_pair(lengthN, seed=None)
            self.A = A1
        elif self.method == "sobol":
            A1, _ = sobol_thresholds_pair(lengthN, seed=None)
            self.A = A1
        else:
            raise ValueError(f"Unknown method {method}")

    def bitstream(self, p_prob: float) -> np.ndarray:
        N = self.lengthN
        val_8bit = int(round(p_prob * 255.0))
        if val_8bit <= 0:
            return np.zeros(N, dtype=np.uint8)
        if val_8bit >= 255:
            return np.ones(N, dtype=np.uint8)

        num = val_8bit * (N - 1)
        rhs = (self.A.astype(np.int64) * 255)
        return (num >= rhs).astype(np.uint8)

class BitstreamGeneratorPair:
    """
    Pair generator for (pixel, weight).
    methods: adus_sdus / sobol / halton / lfsr / random8 / umul / htc / vlsi22
    qmc_mode: fixed vs fresh
    """
    def __init__(self, method: str, lengthN: int, *,
                 sdus_a: int | None = None,
                 lfsr_seed1: int = 0xA5,
                 lfsr_seed2: int = 0xC3,
                 qmc_mode: str = "fixed"):
        self.method = method.lower()
        self.lengthN = lengthN
        self.sdus_a = sdus_a if sdus_a is not None else A_DICT.get(lengthN)
        if self.method == "adus_sdus" and self.sdus_a is None:
            raise ValueError(f"No SDUS a for N={lengthN}")
        if qmc_mode not in ("fixed", "fresh"):
            raise ValueError("qmc_mode must be fixed or fresh")
        self._is_fresh = (qmc_mode == "fresh")

        # caches / engines
        self.pix = None
        self.wgt = None

        self._pair_cache = {"sobol": None, "halton": None, "random8": None}
        self._eng2d_sobol = None
        self._eng2d_halton = None
        self._rng_pair = None

        # uMUL state
        self._umul_A_one = None
        self._umul_A_zero = None
        self._umul_idx_one = 0
        self._umul_idx_zero = 0
        self._eng_pix = None
        self._eng_wgt1 = None
        self._eng_wgt0 = None

        # HTC
        self._htc_sel = None

        if self.method == "adus_sdus":
            self.pix = BitstreamGenerator("adus", lengthN)
            self.wgt = BitstreamGenerator("sdus", lengthN, sdus_a=self.sdus_a)

        elif self.method == "sobol":
            if not self._is_fresh:
                A1, A2 = sobol_thresholds_pair(lengthN, seed=None)
                self.pix = BitstreamGenerator("sobol", lengthN, A_override=A1)
                self.wgt = BitstreamGenerator("sobol", lengthN, A_override=A2)
            else:
                from scipy.stats.qmc import Sobol
                self._eng2d_sobol = Sobol(d=2, scramble=False, seed=None, bits=39)

        elif self.method == "halton":
            if not self._is_fresh:
                A1, A2 = halton_thresholds_pair(lengthN, seed=None)
                self.pix = BitstreamGenerator("halton", lengthN, A_override=A1)
                self.wgt = BitstreamGenerator("halton", lengthN, A_override=A2)
            else:
                from scipy.stats.qmc import Halton
                self._eng2d_halton = Halton(d=2, scramble=False, seed=None, bits=39)

        elif self.method == "lfsr":
            # fixed: two independent seeds
            if not self._is_fresh:
                self.pix = BitstreamGenerator("lfsr", lengthN, lfsr_seed=lfsr_seed1)
                self.wgt = BitstreamGenerator("lfsr", lengthN, lfsr_seed=lfsr_seed2)
            else:
                # For simplicity and to stay faithful: fresh mode re-draw thresholds per call using new seeds derived from RNG.
                self._rng_pair = np.random.default_rng()

        elif self.method == "random8":
            if not self._is_fresh:
                A1, A2 = random8_thresholds_pair(lengthN)
                self.pix = BitstreamGenerator("random8", lengthN, A_override=A1)
                self.wgt = BitstreamGenerator("random8", lengthN, A_override=A2)
            else:
                self._rng_pair = np.random.default_rng()

        elif self.method == "umul":
            # uMUL baseline: Sobol thresholds, weight thresholds depend on pixel stream bits
            if not self._is_fresh:
                A1, A2 = sobol_thresholds_pair(lengthN, seed=None)
                B1, _  = sobol_thresholds_pair(lengthN, seed=None)
                self.pix = BitstreamGenerator("sobol", lengthN, A_override=A1)
                self._umul_A_one  = A2.astype(np.int64)
                self._umul_A_zero = B1.astype(np.int64)
                self._umul_idx_one = 0
                self._umul_idx_zero = 0
            else:
                from scipy.stats.qmc import Sobol
                self._eng_pix  = Sobol(d=1, scramble=False, seed=None, bits=39)
                self._eng_wgt1 = Sobol(d=1, scramble=False, seed=None, bits=39)
                self._eng_wgt0 = Sobol(d=1, scramble=False, seed=None, bits=39)

        elif self.method == "htc":
            assert is_pow2(lengthN), "HTC requires N=2^n"
            self._htc_sel = htc_build_fsm_schedule_sobol(lengthN)

        elif self.method == "vlsi22":
            pass

        else:
            raise ValueError(f"Unknown method {method}")

    def _draw_pair_thresholds(self, which: str):
        N = self.lengthN
        which = which.lower()
        if which == "sobol":
            S = self._eng2d_sobol.random(n=N)
            A1 = np.clip((S[:, 0] * N).astype(int), 0, N - 1).astype(np.int32)
            A2 = np.clip((S[:, 1] * N).astype(int), 0, N - 1).astype(np.int32)
            self._pair_cache["sobol"] = (A1, A2)
        elif which == "halton":
            H = self._eng2d_halton.random(n=N)
            A1 = np.clip((H[:, 0] * N).astype(int), 0, N - 1).astype(np.int32)
            A2 = np.clip((H[:, 1] * N).astype(int), 0, N - 1).astype(np.int32)
            self._pair_cache["halton"] = (A1, A2)
        elif which == "random8":
            R = self._rng_pair.integers(0, 256, size=(2, N), dtype=np.int32)
            A1 = (R[0] * (N - 1) // 255).astype(np.int32)
            A2 = (R[1] * (N - 1) // 255).astype(np.int32)
            self._pair_cache["random8"] = (A1, A2)
        else:
            raise ValueError(f"Unsupported draw {which}")

    def pixel(self, p_prob: float) -> np.ndarray:
        N = self.lengthN
        m = self.method

        if m == "adus_sdus":
            return self.pix.bitstream(p_prob)

        if m == "sobol":
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            if self._pair_cache["sobol"] is None or self._pair_cache["sobol"][0] is None:
                self._draw_pair_thresholds("sobol")
            A1, A2 = self._pair_cache["sobol"]
            self._pair_cache["sobol"] = (None, A2)
            return BitstreamGenerator("sobol", N, A_override=A1).bitstream(p_prob)

        if m == "halton":
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            if self._pair_cache["halton"] is None or self._pair_cache["halton"][0] is None:
                self._draw_pair_thresholds("halton")
            A1, A2 = self._pair_cache["halton"]
            self._pair_cache["halton"] = (None, A2)
            return BitstreamGenerator("halton", N, A_override=A1).bitstream(p_prob)

        if m == "lfsr":
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            # simple fresh redraw (documented in README)
            A = generate_lfsr_sequence_auto(int(self._rng_pair.integers(1, 2**16)), N)
            return BitstreamGenerator("lfsr", N, A_override=A).bitstream(p_prob)

        if m == "random8":
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            if self._pair_cache["random8"] is None or self._pair_cache["random8"][0] is None:
                self._draw_pair_thresholds("random8")
            A1, A2 = self._pair_cache["random8"]
            self._pair_cache["random8"] = (None, A2)
            return BitstreamGenerator("random8", N, A_override=A1).bitstream(p_prob)

        if m == "umul":
            if not self._is_fresh:
                return self.pix.bitstream(p_prob)
            S1 = self._eng_pix.random(n=N)[:, 0]
            A1 = np.clip((S1 * N).astype(int), 0, N - 1).astype(np.int32)
            return BitstreamGenerator("sobol", N, A_override=A1).bitstream(p_prob)

        if m == "htc":
            return htc_encode_TB(p_prob, N)

        if m == "vlsi22":
            return vlsi22_generate_stream_for_value(p_prob, N)

        raise ValueError(f"Unknown method {m}")

    def weight(self, w_prob: float, pixel_stream: np.ndarray | None = None) -> np.ndarray:
        N = self.lengthN
        m = self.method

        if m == "adus_sdus":
            return self.wgt.bitstream(w_prob)

        if m == "sobol":
            if not self._is_fresh:
                return self.wgt.bitstream(w_prob)
            if self._pair_cache["sobol"] is None or self._pair_cache["sobol"][1] is None:
                self._draw_pair_thresholds("sobol")
            A1, A2 = self._pair_cache["sobol"]
            self._pair_cache["sobol"] = None
            return BitstreamGenerator("sobol", N, A_override=A2).bitstream(w_prob)

        if m == "halton":
            if not self._is_fresh:
                return self.wgt.bitstream(w_prob)
            if self._pair_cache["halton"] is None or self._pair_cache["halton"][1] is None:
                self._draw_pair_thresholds("halton")
            A1, A2 = self._pair_cache["halton"]
            self._pair_cache["halton"] = None
            return BitstreamGenerator("halton", N, A_override=A2).bitstream(w_prob)

        if m == "lfsr":
            if not self._is_fresh:
                return self.wgt.bitstream(w_prob)
            A = generate_lfsr_sequence_auto(int(self._rng_pair.integers(1, 2**16)), N)
            return BitstreamGenerator("lfsr", N, A_override=A).bitstream(w_prob)

        if m == "random8":
            if not self._is_fresh:
                return self.wgt.bitstream(w_prob)
            if self._pair_cache["random8"] is None or self._pair_cache["random8"][1] is None:
                self._draw_pair_thresholds("random8")
            A1, A2 = self._pair_cache["random8"]
            self._pair_cache["random8"] = None
            return BitstreamGenerator("random8", N, A_override=A2).bitstream(w_prob)

        if m == "htc":
            return htc_encode_RB_by_FSM(w_prob, N, self._htc_sel)

        if m == "vlsi22":
            return vlsi22_generate_stream_for_value(w_prob, N)

        if m == "umul":
            val8 = int(round(w_prob * 255.0))
            if val8 <= 0:
                return np.zeros(N, dtype=np.uint8)
            if val8 >= 255:
                return np.ones(N, dtype=np.uint8)

            num = val8 * (N - 1)
            rhs_scale = 255
            out = np.empty(N, dtype=np.uint8)
            if pixel_stream is None:
                pixel_stream = np.zeros(N, dtype=np.uint8)

            if not self._is_fresh:
                A1 = self._umul_A_one
                A0 = self._umul_A_zero
                i1 = self._umul_idx_one
                i0 = self._umul_idx_zero
                for i in range(N):
                    if pixel_stream[i] == 1:
                        out[i] = 1 if num >= A1[i1] * rhs_scale else 0
                        i1 = (i1 + 1) % N
                    else:
                        out[i] = 1 if num >= A0[i0] * rhs_scale else 0
                        i0 = (i0 + 1) % N
                self._umul_idx_one = i1
                self._umul_idx_zero = i0
                return out

            # fresh uMUL
            for i in range(N):
                if pixel_stream[i] == 1:
                    t = int(self._eng_wgt1.random(n=1)[0, 0] * N)
                else:
                    t = int(self._eng_wgt0.random(n=1)[0, 0] * N)
                t = min(max(t, 0), N - 1)
                out[i] = 1 if num >= t * rhs_scale else 0
            return out

        raise ValueError(f"Unknown method {m}")