# 3×3 Edge Detection (Sobel) with SC Bitstreams

This folder evaluates a 3×3 Sobel edge detector using stochastic computing (SC) bitstreams.
It compares DUS (ADUS/SDUS) against Sobol/Halton/LFSR/Random8/uMUL and several unary/temporal baselines.

## What it does
- Loads an RGB image and processes the grayscale channel (normalized to [0,1]).
- Uses 3×3 Sobel kernels rescaled to keep weights in [-1,1].
- Computes an EXACT reference in bipolar arithmetic:
  p_bi = 2p-1, then gx = Σ p_bi * w, gy = Σ p_bi * w, and |G| = sqrt(gx^2+gy^2).
- Normalizes magnitude by a provable bound:
  norm_factor = sqrt(||Kx||_1^2 + ||Ky||_1^2), then output = (|G|/norm_factor)*255.

## SC pipeline (bitstream path)
- For each pixel and each kernel weight, generate bitstreams for pixel and weight.
- Multiply in bipolar SC using XNOR.
- Sum multiple products using unary scaled addition (uSADD).
- Compute magnitude and normalize using the same norm_factor.

Correlation mitigation:
- Pixel streams are replicated (tile), weight streams are rotated (roll) before multiplication.
- VLSI'22 (Downscale) uses its own paired construction and does NOT apply replicate/rotate.

## fixed vs fresh generation
Some baselines support:
- `fixed`: one pair of thresholds is generated and reused across the entire image.
- `fresh`: a new threshold pair is generated for each multiplication.
DUS (ADUS/SDUS) is always treated as fixed templates.

## Run
Install deps:
```bash
pip install -r requirements.txt