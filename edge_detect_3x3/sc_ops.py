from __future__ import annotations
import numpy as np

def F_mul(a, b, mode: str = "xnor"):
    if mode == "xnor":
        return np.logical_not(np.logical_xor(a, b)).astype(np.uint8)
    elif mode == "and":
        return np.logical_and(a, b).astype(np.uint8)
    elif mode == "exact":
        return a * b
    else:
        raise ValueError(f"Unsupported multiplication mode: {mode}")

def uSADD_stream(streams: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    """
    Unary scaled addition (streaming):
    - For each cycle t, count ones c_t across N inputs, accumulate acc += c_t.
    - If acc >= N, output 1 and acc -= N, else output 0.
    Output length equals input streams' length.
    """
    assert len(streams) >= 1, "Need at least 1 input stream"
    L = len(streams[0])
    for s in streams:
        assert len(s) == L, "All streams must have the same length"

    S = np.stack([s.astype(np.uint8) for s in streams], axis=0)  # (N, L)
    counts = np.sum(S, axis=0).astype(np.int32)

    out = np.zeros(L, dtype=np.uint8)
    acc = 0
    N = S.shape[0]
    for t in range(L):
        acc += int(counts[t])
        if acc >= N:
            out[t] = 1
            acc -= N
    return out

def uSADD_sum_value_from_streams(streams: list[np.ndarray] | tuple[np.ndarray, ...],
                                 bipolar: bool = True) -> float:
    """
    Convert uSADD output stream to a numeric "true sum":
    - out_stream encodes the mean across inputs (scaled addition).
    - Multiply by number of inputs to recover the (scaled) sum.
    """
    out_stream = uSADD_stream(streams)
    p_mean = float(out_stream.mean())
    if bipolar:
        x_mean = 2.0 * p_mean - 1.0
        return len(streams) * x_mean
    else:
        return len(streams) * p_mean