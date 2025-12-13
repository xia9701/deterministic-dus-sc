This repository provides the open-source evaluation code for our work on deterministic stochastic computing (SC) using Deterministic Unary Sequences (DUS), with a focus on in-memeory SC architectures.

The code covers bitstream-level, operator-level, and task-level evaluations, and compares DUS against conventional stochastic and low-discrepancy baselines.

deterministic-dus-sc/
├── b2s_error/             # Binary-to-stochastic (B2S) conversion error
├── unary_mul_quality/     # Unary multiplication accuracy and correlation（SCC/ZCE/MUL MAE）
├── edge_detecter_3x3/     # Operator-level 3×3 edge detection
└── bsds500_cuda/          # Task-level BSDS500 Canny evaluation (CUDA)

Each subdirectory is self-contained and includes its own README and run scripts.

Evaluation Scope
	•	Bitstream level: B2S conversion accuracy and distribution quality
	•	Arithmetic level: Unary multiplication error under different generators
	•	Operator level: 3×3 convolution-based edge detection
	•	Task level: BSDS500 Canny edge detection (ODS / OIS / AP)
Supported generators include:
	•	DUS (ADUS & SDUS, deterministic)
	•	Sobol, Halton
	•	LFSR, Random8
	•	uMUL (uGEMM)
	•	VLSI’22 Downscale
	•	HTC (ASPDAC’25)
	•	Temporal Unary (TU / TUB2)
	•	Floating-point exact baseline


Reproducibility Notes
	•	Unary precision is fixed (e.g., N=256) unless otherwise specified.
	•	DUS uses fixed deterministic templates, reflecting realistic hardware deployment.
	•	Stochastic baselines support fixed or fresh generation modes where applicable.
	•	Random seeds are controlled when possible and documented per submodule.
Tested Environment
	•	GPU: NVIDIA RTX A6000 × 8
	•	CPU: Intel Xeon Gold 6330 × 2 (56C / 112T)
	•	RAM: 640 GB
	•	Driver: 570.133.20
	•	CUDA: 12.8
	•	Python: 3.9+

GPU execution is recommended for bsds500_cuda.

License and Usage

This code is released for research and academic use.
If you use this repository, please cite the corresponding paper.
