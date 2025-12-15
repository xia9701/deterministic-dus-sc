This repository provides open-source evaluation code for our work on deterministic stochastic computing (SC) based on Deterministic Unary Sequences (DUS), and is designed to be applicable to in-DRAM SC architectures.
The code covers bitstream-, operator-, and task-level evaluations, and compares DUS (including ADUS and SDUS) against conventional stochastic generators (e.g., LFSR), low-discrepancy sequences (Sobol and Halton), and representative recent unary computing approaches.


deterministic-dus-sc/
├── b2s_error/             # Binary-to-stochastic (B2S) conversion error
├── unary_mul_quality/     # Unary multiplication accuracy and correlation（SCC/ZCE/MUL MAE/ADD MAE）
├── edge_detecter_3x3/     # Operator-level 3×3 edge detection
└── bsds500_cuda/          # Task-level BSDS500 Canny evaluation (CUDA)

Each subdirectory is self-contained and includes its own README and run scripts.

Evaluation Scope
	•	Bitstream level: B2S conversion accuracy and distribution quality
	•	Arithmetic level: Unary multiplication error under different generators
	•	Operator level: 3×3 convolution-based edge detection
	•	Task level: BSDS500 Canny edge detection (ODS / OIS / AP)
Baselines include:
	•	DUS (ADUS & SDUS, deterministic)
	•	Random (ideal i.i.d. Bernoulli bitstreams with 50% ‘1’ / 50% ‘0’)
	•	Sobol, Halton，LFSR
	•	Floating-point exact baseline
	•	uGEMM /Micro'21[1]
	•	Downscale/VLSI’22 [2]
	•	tubGEMM/ ISVLSI'23[3]
	•	HTC/ASPDAC’25 [4]
	
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


Reference:
[1] D. Wu, J. Li, R. Yin, H. Hsiao, Y. Kim, and J. San Miguel, “ugemm: Unary computing for gemm applications,” IEEE Micro, vol. 41, no. 3, pp. 50–56, 2021.
[2]	Y. Kiran and M. Riedel, “A scalable, deterministic approach to stochastic computing,” in Proceedings of the Great Lakes Symposium on VLSI 2022, 2022, pp. 45–51.
[3]	P. Vellaisamy, H. Nair, J. Finn, M. Trivedi, A. Chen, A. Li, T.-H. Lin, P. Wang, S. Blanton, and J. P. Shen, “tubgemm: Energy-efficient and sparsity-effective temporal-unary-binary based matrix multiply unit,” in 2023 IEEE Computer Society Annual Symposium on VLSI (ISVLSI). IEEE, 2023, pp. 1–6.
[4] M. Tasnim, S. Sachdeva, Y. Liu, and S. X. Tan, “Hybrid temporal computing for lower power hardware accelerators,” in Proceedings of the 30th Asia and South Pacific Design Automation Conference, 2025, pp. 237–244.


License and Usage

This code is released for research and academic use.
If you use this repository, please cite the corresponding paper.
