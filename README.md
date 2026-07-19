# 🔐 Cryptographic Forgery in E-Contracts
### A Survey-Based Implementation of Metaheuristic Length-Extension Attack Detection & Blockchain Anchoring

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![ML](https://img.shields.io/badge/ML-Dynamic_Studio-green?style=for-the-badge&logo=scikit-learn)
![Blockchain](https://img.shields.io/badge/Ledger-Anchored-blueviolet?style=for-the-badge&logo=blockchaindotcom)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 About This Project

This project is a **full-stack working implementation** based on our survey paper:

> **"Cryptographic Forgery in E-Contracts: A Survey of Metaheuristic Length-Extension Attacks"**
> Department of Computer Science and Engineering, JAIN Deemed-to-be University, Bangalore, India

The system demonstrates how **hash length-extension attacks** work on electronic contracts across **MD5, SHA-1, and SHA-256**,
how **Simulated Annealing (metaheuristic optimization)** is used to guess secret key lengths, and how a multi-layered defense (combining **Random Forest ML classifiers**, **hmac**, and **Proof-of-Work blockchain anchoring**) secures contract payloads — all through a premium dark glassmorphism security dashboard.

---

## 🤖 AI Tools Used in This Project

This project was built using a combination of AI assistants, each contributing to different parts of the work.

---

### 🟠 Claude (Anthropic) — Original Implementation
**Used for:** Initial mock code templates and frontend scaffolding.

---

### 🟢 Google Antigravity (DeepMind) — Core Upgrades & Cryptographic Engine
**Used for:** Authentic cryptographic engineering, ML Model Studio, and blockchain ledger integration.

Antigravity refactored the project to feature:
- A functional pure-Python **cryptographic compression engine** performing actual length-extension signature hijacking.
- An interactive **ML Model Studio** for training multiple classifiers (Random Forest, Decision Tree, Logistic Regression) dynamically on the backend and drawing live ROC and Confusion Matrix SVG charts in the frontend.
- An in-memory **Proof-of-Work blockchain ledger** that anchors contract hashes, computes Merkle roots, and simulates mining hashes at custom difficulties.
- A **multi-layered verification console** checking cryptographic signatures, ML anomaly flags, and ledger registration simultaneously.

---

### 🔵 Google Gemini — Research Assistance
**Used for:** Literature search, paper summaries, and content structuring.

Gemini was used to:
- Search and summarize academic papers related to hash length-extension attacks.
- Explain the Merkle–Damgård construction in simple terms.
- Find recent papers (2020–2025) from IEEE, ACM, and ScienceDirect.
- Suggest relevant legal frameworks (GDPR, IT Act 2000, UNCITRAL).
- Help understand how Simulated Annealing applies to cryptographic search problems.

---

### 🟣 ChatGPT (OpenAI) — Survey Paper Writing & Explanation
**Used for:** Writing sections of the survey paper and explaining concepts.

ChatGPT was used to:
- Draft and refine sections of the survey paper (Abstract, Introduction, Problem Statement).
- Explain cryptographic concepts in plain language for the presentation.
- Suggest the structure of the PRISMA-based literature review.
- Write the comparative analysis narrative for Table II.

---

## 🧠 What the System Does

### Core Concepts from the Survey Paper

| Concept | What It Means | Where in Code |
|---------|--------------|---------------|
| Hash Length-Extension Attack | Appending data to a signed contract without knowing the secret key | `detector.py → length_extension_attack()` |
| Merkle–Damgård Vulnerability | MD5/SHA-1/SHA-256 internal chaining registers can be resumed by attacker | `detector.py → sha256_compress(), md5_compress(), sha1_compress()` |
| Simulated Annealing | Metaheuristic optimizer that guesses the secret key length | `detector.py → simulated_annealing_secret_length()` |
| HMAC Double Hashing | Secure MAC nested double-hashing that blocks length extension | `detector.py → secure_sign()` |
| Random Forest / ML | Classifier that detects hidden padding bytes in forged payloads | `detector.py → train_custom_classifier()` |
| Blockchain Anchoring | Immutable Proof-of-Work ledger securing signature histories | `app.py → Blockchain Class` |
| Multi-layered Audit | Unified validation combining signature, ML, and blockchain checks | `app.py → verify()` |

---

## 🗂️ Project Structure

```
econtract_security/
│
├── app.py                  ← Flask backend (all API routes & Blockchain ledger)
├── detector.py             ← Core logic: custom crypto compressions, SA, ML models
├── requirements.txt        ← Python dependencies
│
└── templates/
    └── index.html          ← Premium dark glassmorphic UI, SVG charts, visual miners
```

---

## ⚙️ How to Run

### Step 1 — Make sure Python is installed
```bash
python --version
# Should show Python 3.8 or higher
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the server
```bash
python app.py
```

### Step 4 — Open in browser
```
http://127.0.0.1:5000
```

---

## 🖥️ Dashboard Features

| Page | What It Does |
|------|-------------|
| 🏠 Dashboard | System overview, active parameters, and interactive guides |
| ✍️ Sign Contract | Generate vulnerable raw signatures or secure HMACs (MD5, SHA-1, SHA-256) |
| ⚠️ Attack Sandbox | Launch a cryptographic length-extension exploit using SA guesses |
| 🔍 Detect Forgery | Scan any transmitted payload for binary padding anomalies |
| 🧬 ML Model Studio | Train Random Forest, Decision Trees, or Logistic Regression live with SVG plots |
| ⛓️ Blockchain Ledger | Anchor contract signatures and mine PoW blocks with difficulty adjustments |
| 🛡️ Verify Console | Check contracts via Signature Match, ML detectors, and Ledger status |
| 📊 Survey Table | View the full taxonomy map and comparative data from the survey paper |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sign/vulnerable` | Sign message with raw vulnerable hashing |
| POST | `/api/sign/secure` | Sign message with secure HMAC double-hashing |
| POST | `/api/attack/length-extension` | Run actual cryptographic length-extension using state reconstruction |
| POST | `/api/detect` | Run active classifier anomaly check on contract data |
| POST | `/api/verify` | Perform multi-layered validation (Signature, ML, Blockchain) |
| POST | `/api/ml/train` | Train a customized ML model with user hyperparameters |
| POST | `/api/blockchain/anchor` | Submit signature tag to the pending transactions pool |
| POST | `/api/blockchain/mine` | Run nonce search to mine pending block at set difficulty |
| GET  | `/api/blockchain/blocks` | List mined blocks chain and transactions |
| GET  | `/api/stats` | Retrieve classifier and active engine parameters |

---

## 📊 ML Model Details

- **Classifiers Supported:** Random Forest, Decision Tree, Logistic Regression (scikit-learn)
- **Training dataset:** 1,200 samples (600 legitimate + 600 forged synthetic contract texts)
- **Features used for detection:**

| # | Feature | Why It Matters |
|---|---------|---------------|
| 1 | Payload length | Forged payloads tend to be longer due to padding |
| 2 | Padding byte detected | Presence of 0x80 byte indicates MD padding injection |
| 3 | Block alignment | Forged payloads align to 64-byte boundaries |
| 4 | Tag entropy | Forged tags show statistical randomness deviations |
| 5 | Null byte ratio | MD padding contains multiple 0x00 null bytes |
| 6 | Length mod 512 | Block structure byte length anomaly detection |
| 7 | Average byte value | Average ASCII code shifted by null/padding bytes |
| 8 | Payload entropy | Legitimate English payloads show higher character diversity |

---

## 🔬 Simulated Annealing in This Project

The SA algorithm is used to **guess the secret key length** without knowing the key itself.

```
Initialize random secret length guess
Set temperature T = 100

Repeat 300 iterations:
    Generate neighbour guess (±1 or ±2)
    Score guess based on block alignment heuristic
    If better → accept
    If worse  → accept with probability exp(ΔScore / T)
    Reduce T by factor 0.95

Return best guess
```

This mirrors how a real attacker would search for the correct secret length before rebuilding the padding bytes and hijacking the hashing chaining variables.

---

## 📚 Papers Referenced in This Project

| # | Paper | Year | Source |
|---|-------|------|--------|
| 1 | Bellare et al. — Keying Hash Functions for Message Authentication | 1996 | CRYPTO |
| 2 | NIST FIPS 180-4 — Secure Hash Standard | 2015 | NIST |
| 3 | Kirkpatrick et al. — Optimization by Simulated Annealing | 1983 | Science |
| 4 | Goldberg — Genetic Algorithms in Search and Optimization | 1989 | Book |
| 5 | Kennedy & Eberhart — Particle Swarm Optimization | 1995 | IEEE |
| 6 | Frank DENIS — Length-Extension Attacks Are Still a Thing | 2025 | Blog |
| 7 | Enhancing IDS Using Metaheuristic Algorithms | 2024 | DJES |
| 8 | Metaheuristic Feature Selection for Cyberattack Detection | 2025 | Scientific Reports |
| 9 | Comprehensive Review of AI-Driven Detection Techniques | 2024 | Journal of Big Data |

---

## ⚖️ Legal Frameworks Covered

| Framework | Relevant Sections | What It Means for E-Contract Forgery |
|-----------|------------------|--------------------------------------|
| GDPR (EU) | Articles 5, 25, 32 | Organizations must use secure MACs (HMAC/SHA3) by design |
| IT Act 2000 (India) | Sections 43, 66, 73, 74 | Hash/Signature forgery is a criminal cyber offense |
| UNCITRAL Model Law | Articles 8, 9, 13 | Forged contracts fail integrity checks and are legally void |

---

## 🏫 Academic Context

- **Paper Title:** Cryptographic Forgery in E-Contracts: A Survey of Metaheuristic Length-Extension Attacks
- **Institution:** JAIN Deemed-to-be University, Bangalore, India
- **Department:** Computer Science and Engineering
- **Methodology:** PRISMA Systematic Literature Review
- **Papers Reviewed:** 54 peer-reviewed publications (2018–2025)
- **Databases Used:** IEEE Xplore, ACM Digital Library, ScienceDirect

---

## 👥 Team

| Name | Role |
|------|------|
| Vedanth D | Research, Implementation, Survey |
| Mursalin Pasha M | Research, Survey Writing |

---

## 📝 License

This project is for **academic and educational purposes only.**
The attack simulation is a controlled demonstration — do not use against real systems.
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

```
MIT License — Free to use for learning and research
```

---

<div align="center">
  <b>Built with understanding the cryptography technics.</b><br/>
  <i>Cyber Law · Computer Science · Cryptographic Security</i>
</div>
