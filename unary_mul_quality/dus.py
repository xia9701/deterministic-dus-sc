#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import gcd


def generate_deterministic_uniform_sequence(N, a):
    """DUS: [(a*i) mod N], require gcd(a,N)=1."""
    assert gcd(a, N) == 1
    return [((a * i) % N) for i in range(N)]