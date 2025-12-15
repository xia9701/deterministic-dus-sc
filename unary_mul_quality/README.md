# Unary Multiplication & Bitstream Quality (MAE / SCC / ZCE)

This folder evaluates deterministic unary sequences (DUS: ADUS/SDUS) against
common baselines (Random, LFSR, Halton, Sobol) using Monte Carlo trials.

## What this experiment does
- Sweeps bitstream length N in {16, 32, 64, 128, 256, 512, 1024}
- Uses 8-bit inputs, mapped to unary thresholds
- Reports:
  - Unary multiplication MAE (bitwise AND)
  - Scaled-add MAE using a MUX adder (reference: (A+B)/2)
  - Stochastic cross-correlation (SCC)
  - Zero-correlation error (ZCE)

## How to run
From the repository root:
```bash
pip install -r requirements.txt
python3 run_unary_eval.py