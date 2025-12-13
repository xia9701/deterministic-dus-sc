# metrics.py
# B2S error metric 

from __future__ import annotations


def b2s_error_from_stream(stream: str, target: float, lengthN: int) -> float:
    """
      error = abs( (sum(bits)/(lengthN - 1)) - target )
    """
    ones = sum(1 for b in stream if b == "1")
    return abs((ones / (lengthN - 1)) - target)