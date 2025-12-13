# B2S Error Sweep (Random / LFSR / Halton / Sobol / uMUL / DUS)

This folder evaluates binary-to-stochastic (B2S) conversion error across bitstream
lengths N in {16, 32, 64, 128, 256, 512, 1024}. It compares deterministic unary
sequences (DUS: ADUS/SDUS) against common baselines.

## What this experiment measures
For each N, the script runs Monte Carlo trials with random n-bit inputs where:
- BIT_WIDTH = log2(N)
- MAX_INT = 2^BIT_WIDTH - 1
- Each method generates unary streams using threshold comparison
- B2S error is computed as the mean of A/B absolute errors

Metrics:
- B2S error per input: | (#ones/(N-1)) - target |
- Reported value: average over trials and over A/B

## Run
From the repository root:
```bash
pip install -r requirements.txt
python b2s_error/run_b2s_sweep.py