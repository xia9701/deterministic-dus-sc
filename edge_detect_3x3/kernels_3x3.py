from __future__ import annotations
import numpy as np

def get_kernels_3x3(name: str = "sobel"):
    """
    Return (Kx, Ky, norm_factor).
    - We rescale Sobel/Scharr to keep weights in [-1,1] for bipolar SC.
    - norm_factor = sqrt(||Kx||_1^2 + ||Ky||_1^2), used to normalize |G| to 0..255.
    """
    name = name.lower()
    if name == "sobel":
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=float)
        Ky = np.array([[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]], dtype=float)
        scale = 2.0
        Kx /= scale
        Ky /= scale
    elif name == "scharr":
        Kx = np.array([[-3, 0, 3],
                       [-10, 0, 10],
                       [-3, 0, 3]], dtype=float)
        Ky = np.array([[ 3, 10,  3],
                       [ 0,  0,  0],
                       [-3,-10, -3]], dtype=float)
        scale = 10.0
        Kx /= scale
        Ky /= scale
    else:
        raise ValueError("Unknown kernel set")

    L1x = float(np.abs(Kx).sum())
    L1y = float(np.abs(Ky).sum())
    norm = np.sqrt(L1x**2 + L1y**2)
    return Kx, Ky, norm