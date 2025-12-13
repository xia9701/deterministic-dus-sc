# BSDS500 Canny Evaluation (CUDA / CuPy)

This directory contains the task-level evaluation code for the BSDS500
benchmark using a Canny edge detection pipeline, where **only the gradient
computation stage is implemented using stochastic computing (SC)**.

The implementation supports both **GPU (CuPy/CUDA)** and **CPU (NumPy/SciPy)**
execution. For full BSDS500 evaluation, GPU execution is strongly recommended.

---

## Overview

The evaluation follows a standard Canny pipeline:

1. Gaussian smoothing  
2. Gradient computation (FP or SC-based Sobel)  
3. Non-maximum suppression (NMS)  
4. Double thresholding and hysteresis  
5. BSDS500 boundary evaluation (ODS / OIS / AP)

Only **Step 2 (gradient computation)** differs across methods; all subsequent
stages are shared to ensure a fair task-level comparison.

The following methods are supported:

- **DUS (ADUS & SDUS)** *(proposed, deterministic)*
- Sobol (2D low-discrepancy sequence)
- Halton
- LFSR
- Random8
- uGEMM (uMUL)
- VLSI’22 Downscale
- HTC (ASPDAC’25)
- Temporal Unary (TU / TUB2)
- Floating-point exact baseline

The code reports both:
- **Operator-level accuracy** (e.g., MAE against FP Sobel), and
- **Task-level BSDS500 metrics**, including:
  - ODS (Optimal Dataset Scale)
  - OIS (Optimal Image Scale)
  - AP (Average Precision from PR curves)

---

## Hardware and Software Environment

The experiments were validated on the following platform:

- **GPU**: NVIDIA RTX A6000 × 8  
- **CPU**: Intel Xeon Gold 6330 × 2 (56 cores / 112 threads total)  
- **Memory**: 640 GB RAM  
- **NVIDIA Driver**: 570.133.20  
- **CUDA**: 12.8  
- **Python**: 3.9+ recommended  

The code can run on CPU-only systems, but the full BSDS500 evaluation
is computationally intensive and **CUDA acceleration is strongly recommended**.

---

## Installation

```bash
cd deterministic-dus-sc/bsds500_cuda

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt