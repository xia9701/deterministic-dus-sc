#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def calc_b2s_error(stream: str, target: float, lengthN: int) -> float:
    """
      stream_int = [int(bit) for bit in stream]
      abs(sum(stream_int)/(lengthN-1) - target)
    """
    stream_int = [int(bit) for bit in stream]
    return abs((sum(stream_int) / (lengthN - 1)) - target)