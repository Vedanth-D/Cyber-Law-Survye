# 🔐 Cryptographic Forgery in E-Contracts
### A Survey-Based Implementation of Metaheuristic Length-Extension Attack Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![ML](https://img.shields.io/badge/ML-RandomForest-green?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 About This Project

This project is a **full-stack working implementation** based on our survey paper:

> **"Cryptographic Forgery in E-Contracts: A Survey of Metaheuristic Length-Extension Attacks"**
> Department of Computer Science and Engineering, JAIN Deemed-to-be University, Bangalore, India

The system demonstrates how **hash length-extension attacks** work on electronic contracts,
how **Simulated Annealing (metaheuristic optimization)** is used to guess secret key lengths,
and how a **Random Forest ML classifier** detects forged contract payloads — all through a
dark-themed interactive security dashboard.

---

## 🤖 AI Tools Used in This Project

This project was built using a combination of AI assistants, each contributing to different parts of the work.

---

### 🟠 Claude (Anthropic) — Code Implementation
**Used for:** All code generation, system architecture, and full-stack development

Claude was used to write every line of code in this project including:

- `app.py` — Flask backend with all REST API endpoints
- `detector.py` — Core cryptographic logic, Simulated Annealing optimizer, Random Forest ML model, feature extraction pipeline
- `templates/index.html` — Complete dark-themed security dashboard frontend with HTML, CSS, and JavaScript
- Debugging and fixing all runtime errors (405 Method Not Allowed, 500 template errors, CORS issues)
- Designing the 8-feature ML detection pipeline
- Writing the SA convergence loop and MD padding calculator

> **Claude was the primary coding assistant for this entire project.**
> Model used: Claude Sonnet (claude.ai)

---

### 🔵 Google Gemini — Research Assistance
**Used for:** Literature search, paper summaries, and content structuring

Gemini was used to:

- Search and summarize academic papers related to hash length-extension attacks
- Explain the Merkle–Damgård construction in simple terms
- Find recent papers (2020–2025) from IEEE, ACM, and ScienceDirect
- Suggest relevant legal frameworks (GDPR, IT Act 2000, UNCITRAL)
- Help understand how Simulated Annealing applies to cryptographic search problems
- Summarize papers that were behind paywalls

> Model used: Gemini 1.5 Pro (gemini.google.com)

---

### 🟢 ChatGPT (OpenAI) — Survey Paper Writing & Explanation
**Used for:** Writing sections of the survey paper and explaining concepts

ChatGPT was used to:

- Draft and refine sections of the survey paper (Abstract, Introduction, Problem Statement)
- Explain cryptographic concepts in plain language for the presentation
- Suggest the structure of the PRISMA-based literature review
- Write the comparative analysis narrative for Table II
- Help phrase the legal-technical mapping section (GDPR Articles, IT Act Sections)
- Proofread and improve academic writing quality

> Model used: ChatGPT-4o (chat.openai.com)

---

### 🟡 Summary of AI Contributions

| Task | Tool Used |
|------|-----------|
| All backend Python code | Claude (Anthropic) |
| All frontend HTML/CSS/JS | Claude (Anthropic) |
| Bug fixing & debugging | Claude (Anthropic) |
| System architecture design | Claude (Anthropic) |
| Literature search & paper summaries | Google Gemini |
| Finding IEEE/ACM papers | Google Gemini |
| Survey paper writing | ChatGPT (OpenAI) |
| Academic language & proofreading | ChatGPT (OpenAI) |
| Concept explanations for PPT | ChatGPT (OpenAI) |
| Research topic ideation | All three tools |

---

## 🧠 What the System Does

### Core Concepts from the Survey Paper

| Concept | What It Means | Where in Code |
|---------|--------------|---------------|
| Hash Length-Extension Attack | Appending data to a signed contract without knowing the secret key | `detector.py → length_extension_attack()` |
| Merkle–Damgård Vulnerability | SHA-256/MD5/SHA-1 internal state can be resumed by attacker | `detector.py → md_padding()` |
| Simulated Annealing | Metaheuristic optimizer that guesses the secret key length | `detector.py → simulated_annealing_secret_length()` |
| HMAC-SHA256 | Secure MAC that prevents length extension | `detector.py → secure_sign()` |
| Random Forest | ML classifier that detects forged contract payloads | `detector.py → train_detector()` |
| 8-Feature Detection | Payload length, padding byte, entropy, alignment, etc. | `detector.py → extract_features()` |

---

## 🗂️ Project Structure

```
econtract_security/
│
├── app.py                  ← Flask backend (all API routes)
├── detector.py             ← Core logic: crypto, SA, ML model
├── requirements.txt        ← Python dependencies
│
└── templates/
    └── index.html          ← Full dark dashboard frontend
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
| 🏠 Dashboard | System overview, architecture, and how-to-use guide |
| ✍️ Sign Contract | Sign a message with vulnerable SHA-256 or secure HMAC-SHA256 |
| ⚠️ Simulate Attack | Launch a length-extension attack using SA to guess secret length |
| 🔍 Detect Forgery | Run the Random Forest ML model on any payload + tag |
| ✔️ Verify Contract | Check if a MAC tag is still valid server-side |
| 🗂️ Taxonomy | View the full hierarchical classification from the survey |
| 📊 Comparison | View the full comparative table from the survey paper |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sign/vulnerable` | Sign with raw SHA-256 (vulnerable) |
| POST | `/api/sign/secure` | Sign with HMAC-SHA256 (secure) |
| POST | `/api/attack/length-extension` | Simulate a length-extension attack with SA |
| POST | `/api/detect` | Run ML forgery detection on a payload |
| POST | `/api/verify` | Verify a MAC tag server-side |
| GET  | `/api/stats` | Get model and system statistics |

---

## 📊 ML Model Details

- **Algorithm:** Random Forest (100 estimators)
- **Training samples:** 1,200 (600 legitimate + 600 forged synthetic)
- **Features used for detection:**

| # | Feature | Why It Matters |
|---|---------|---------------|
| 1 | Payload length | Forged payloads tend to be longer |
| 2 | Padding byte detected | 0x80 byte indicates MD padding was appended |
| 3 | Block alignment | Forged payloads align to 64-byte boundaries |
| 4 | Tag entropy | Forged tags show lower entropy |
| 5 | Null byte ratio | MD padding contains null bytes |
| 6 | Length mod 512 | Block structure anomaly detection |
| 7 | Average byte value | Statistical deviation from legitimate traffic |
| 8 | Payload entropy | Legitimate payloads have higher randomness |

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

This mirrors how a real attacker would use SA to efficiently search for the
correct secret length before crafting the forged MAC tag.

---

## 📚 Papers Referenced in This Project

| # | Paper | Year | Source |
|---|-------|------|--------|
| 1 | Bellare et al. — Keying Hash Functions for Message Authentication | 1996 | CRYPTO |
| 2 | NIST FIPS 180-4 — Secure Hash Standard | 2015 | NIST |
| 3 | NIST FIPS 202 — SHA-3 Standard | 2015 | NIST |
| 4 | Kirkpatrick et al. — Optimization by Simulated Annealing | 1983 | Science |
| 5 | Goldberg — Genetic Algorithms in Search and Optimization | 1989 | Book |
| 6 | Kennedy & Eberhart — Particle Swarm Optimization | 1995 | IEEE |
| 7 | Wikipedia — Length Extension Attack | 2025 | Online |
| 8 | Frank DENIS — Length-Extension Attacks Are Still a Thing | 2025 | Blog |
| 9 | Enhancing IDS Using Metaheuristic Algorithms | 2024 | DJES |
| 10 | Metaheuristic Feature Selection for Cyberattack Detection | 2025 | Scientific Reports |
| 11 | Comprehensive Review of AI-Driven Detection Techniques | 2024 | Journal of Big Data |
| 12 | Systematic Review: Metaheuristics for IIoT Attack Detection | 2026 | AI Review |

---

## ⚖️ Legal Frameworks Covered

| Framework | Relevant Sections | What It Means for E-Contract Forgery |
|-----------|------------------|--------------------------------------|
| GDPR (EU) | Articles 5, 25, 32 | Organizations must use secure MACs by law |
| IT Act 2000 (India) | Sections 43, 66, 73, 74 | Hash forgery is a criminal offense |
| UNCITRAL Model Law | Articles 8, 9, 13 | Forged contracts may be legally void |

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

```
MIT License — Free to use for learning and research
```

---

## 🙏 Acknowledgements

We acknowledge the use of the following AI tools in the development of this project:

- **Claude by Anthropic** — for all code generation and system implementation
- **Google Gemini** — for research assistance and paper discovery
- **ChatGPT by OpenAI** — for survey writing and academic content

> *This project was developed as part of a Cyber Law course at JAIN Deemed-to-be University.*

---

<div align="center">
  <b>Built with ❤️ at JAIN Deemed-to-be University, Bangalore</b><br/>
  <i>Cyber Law · Computer Science · Cryptographic Security</i>
</div>
