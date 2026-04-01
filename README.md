# Source Code Package
## Cryptographic Forgery in E-Contracts: A Survey of Metaheuristic Length-Extension Attacks
### JAIN Deemed-to-be University | Dept. of CSE | Cyber Law Survey 2024-25

---

## Overview

This package contains the complete source code implementation supporting the survey paper.
All modules are self-contained pure Python (NumPy only) — no deep learning framework required.

---

## File Structure

```
src/
├── 01_length_extension_attack.py   # Module 1: Attack simulation + HMAC defense
├── 02_hash_anomaly_detector.py     # Module 2: Feature extraction + anomaly detection
├── 03_metaheuristic_optimizer.py   # Module 3: PSO and GWO threshold optimization
├── 04_lstm_autoencoder.py          # Module 4: LSTM autoencoder forgery detector
├── 05_hmm_state_analyzer.py        # Module 5: HMM e-contract state sequence analysis
├── 06_evaluation_pipeline.py       # Module 6: Full comparative evaluation pipeline
└── README.md                       # This file
```

---

## Module Descriptions

### Module 1 — Length-Extension Attack (`01_length_extension_attack.py`)
Demonstrates the structural vulnerability of Merkle-Damgård SHA-256 to
length-extension attacks in e-contract signing contexts.
- `EContractForger.sign_contract()` — vulnerable raw SHA-256 signing
- `EContractForger.perform_extension_attack()` — appends malicious clause
- `SecureEContractSigner` — HMAC-SHA256 defense (immune to extension)

**Corresponds to:** Section IV.A–B (Mathematical Foundations)

---

### Module 2 — Hash Anomaly Detector (`02_hash_anomaly_detector.py`)
Feature extraction and threshold-based anomaly detection.
- Shannon entropy, padding entropy, null-byte ratio, block-alignment features
- `StaticThresholdDetector` — Category B1 (baseline)
- `EntropyAnomalyDetector` — Category B2c (dynamic, learned thresholds)

**Corresponds to:** Section IV.C, Taxonomy Category B

---

### Module 3 — Metaheuristic Optimizer (`03_metaheuristic_optimizer.py`)
PSO and GWO for detection threshold optimization.
- Fitness function: `F(θ) = w₁·DR(θ) − w₂·FPR(θ) − w₃·Cost(θ)`
- `PSOOptimizer` — 30 particles, inertia weight w=0.729, c₁=c₂=1.494
- `GWOOptimizer` — 30 wolves, α linearly decreasing from 2→0

**Corresponds to:** Section IV.D, Table III (Computational Complexity)

---

### Module 4 — LSTM Autoencoder (`04_lstm_autoencoder.py`)
Unsupervised LSTM autoencoder for forgery detection.
- Pure NumPy LSTM cell implementation (no PyTorch/TensorFlow)
- Training: minimize MSE on legitimate contracts only
- Anomaly score: `score(X) = ||X − Decoder(Encoder(X))||²`
- Threshold: `μ_legit + k·σ_legit` (k=2.0 default)

**Corresponds to:** Section IV.E, Table II (Rao et al. approach)

---

### Module 5 — HMM State Analyzer (`05_hmm_state_analyzer.py`)
Hidden Markov Model for e-contract workflow anomaly detection.
- 6 hidden states: DRAFT_SUBMISSION → HASH_COMPUTATION → SIGNATURE →
  [PAYLOAD_EXTENSION] → VERIFICATION → ARCHIVAL
- Baum-Welch training (EM algorithm) on legitimate sequences
- Viterbi decoding for forensic state path reconstruction
- Anomaly score: log P(O|λ) — lower = more anomalous

**Corresponds to:** Section IV.G, Taxonomy Category B2d

---

### Module 6 — Evaluation Pipeline (`06_evaluation_pipeline.py`)
Unified pipeline running and comparing all detectors.
- Generates synthetic dataset (500 legitimate + 500 forged contracts)
- Evaluates: HMAC Verifier, Static Threshold, Entropy Detector,
  Linear SVM, Random Forest, PSO-Optimized Entropy Detector
- Outputs DR%, FPR%, Accuracy for each method
- Demonstrates attack and HMAC defense end-to-end

**Corresponds to:** Section VI (Comparative Analysis), Table II

---

## Requirements

```
python >= 3.8
numpy >= 1.21
```

Install:
```bash
pip install numpy
```

---

## Running the Code

Run any module directly:
```bash
python 01_length_extension_attack.py
python 02_hash_anomaly_detector.py
python 03_metaheuristic_optimizer.py
python 04_lstm_autoencoder.py
python 05_hmm_state_analyzer.py
python 06_evaluation_pipeline.py   # Recommended: runs full comparison
```

---

## Key Results (from Module 6)

| Method                          |  DR%  | FPR%  |  Acc% |
|---------------------------------|-------|-------|-------|
| HMAC Verifier (Protocol)        | 100.0 |  1.0  |  99.5 |
| Static Threshold (B1)           |  ~72  |  ~18  |  ~77  |
| Entropy Anomaly Detector (B2c)  |  ~91  |  ~4.5 |  ~93  |
| Linear SVM Classifier (C1)      |  ~94  |  ~3.5 |  ~95  |
| Random Forest (C1)              |  ~95  |  ~3.2 |  ~96  |
| PSO-Optimized Entropy (B2c+PSO) |  ~94  |  ~3.0 |  ~96  |

Results confirm the paper's comparative findings (Table II, Section VI).

---

## Connection to Paper Taxonomy

```
Paper Taxonomy Category          Module(s)
─────────────────────────────────────────────────────────
A. Protocol-Level Hardening      Module 1 (HMAC defense)
B1. Static Threshold Methods     Module 2
B2c. Entropy Analysis            Modules 2, 3, 6
B2d. HMM State Modeling          Module 5
C1. Supervised Learning (SVM/RF) Module 6
C2. Deep Learning (LSTM)         Module 4
PSO/GWO Optimization             Module 3
```

---

## Authors
Arjun Krishnamurthy, Divya Raghavan, Santhosh Pillai  
Department of Computer Science and Engineering  
JAIN Deemed-to-be University, Bangalore, India  
Academic Year 2024–25
