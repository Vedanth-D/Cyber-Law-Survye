"""
=============================================================================
Cryptographic Forgery in E-Contracts: A Survey of Metaheuristic
Length-Extension Attacks
-----------------------------------------------------------------------------
Module 2: Hash Integrity Anomaly Detector
=============================================================================

Implements a feature-based anomaly detection framework for identifying
length-extension attack attempts in e-contract submission streams.
Features: payload entropy, padding structure, length deviation, hash state.

References:
  [15] Rao et al., "LSTM-based temporal sequence modeling," Computers & Security 2023
  [33] Kumar et al., "Shannon entropy analysis for hash anomaly detection," 2023
"""

import hashlib
import math
import struct
import os
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─── Feature Extraction ───────────────────────────────────────────────────────

@dataclass
class ContractFeatures:
    """Feature vector extracted from a submitted contract + hash pair."""
    payload_length:       int     # Total byte length of submitted contract
    entropy:              float   # Shannon entropy of contract bytes
    padding_entropy:      float   # Shannon entropy of last 64 bytes (padding region)
    null_byte_ratio:      float   # Fraction of 0x00 bytes (inflated in MD padding)
    high_byte_ratio:      float   # Fraction of bytes ≥ 0x80 (0x80 marker byte)
    block_boundary_flag:  int     # 1 if length is multiple of 64 (suspicious)
    length_mod_64:        int     # payload_length mod 64
    hash_hex_entropy:     float   # Entropy of the hash hex string
    is_forged:            int     # Ground truth label (0=legitimate, 1=forged)


def shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy (bits per byte) of a byte sequence."""
    if not data:
        return 0.0
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    n = len(data)
    return -sum((c/n) * math.log2(c/n) for c in freq.values() if c > 0)


def extract_features(contract: bytes, signature: str, label: int = 0) -> ContractFeatures:
    """
    Extract anomaly detection features from a (contract, signature) pair.

    Args:
        contract:  Contract payload bytes.
        signature: Hex-encoded hash signature.
        label:     Ground-truth label (0=legitimate, 1=forged).

    Returns:
        ContractFeatures instance.
    """
    n = len(contract)
    padding_region = contract[-64:] if n >= 64 else contract

    return ContractFeatures(
        payload_length      = n,
        entropy             = shannon_entropy(contract),
        padding_entropy     = shannon_entropy(padding_region),
        null_byte_ratio     = contract.count(0x00) / max(n, 1),
        high_byte_ratio     = sum(1 for b in contract if b >= 0x80) / max(n, 1),
        block_boundary_flag = int(n % 64 == 0),
        length_mod_64       = n % 64,
        hash_hex_entropy    = shannon_entropy(signature.encode()),
        is_forged           = label,
    )


def features_to_vector(f: ContractFeatures) -> np.ndarray:
    return np.array([
        f.payload_length,
        f.entropy,
        f.padding_entropy,
        f.null_byte_ratio,
        f.high_byte_ratio,
        f.block_boundary_flag,
        f.length_mod_64,
        f.hash_hex_entropy,
    ], dtype=np.float64)


# ─── Dataset Generator ───────────────────────────────────────────────────────

def md_padding(message_length: int) -> bytes:
    """Merkle-Damgård padding for SHA-256."""
    bit_len = message_length * 8
    pad = b'\x80'
    pad_len = (56 - (message_length + 1) % 64) % 64
    pad += b'\x00' * pad_len
    pad += struct.pack('>Q', bit_len)
    return pad


def generate_dataset(
    n_legitimate: int = 500,
    n_forged: int = 500,
    secret_key: bytes = b"JAIN_KEY_2024",
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a labeled dataset of legitimate and length-extended forged contracts.

    Args:
        n_legitimate: Number of legitimate contract samples.
        n_forged:     Number of forged (length-extended) samples.
        secret_key:   Secret used for signing.
        seed:         Random seed for reproducibility.

    Returns:
        (X, y) where X is (n_samples, n_features) and y is (n_samples,) labels.
    """
    random.seed(seed)
    np.random.seed(seed)

    contract_templates = [
        b"CONTRACT_ID: {id}\nPARTIES: {p1}, {p2}\nPAYMENT: INR {amt}\nJURISDICTION: Bangalore\n",
        b"AGREEMENT: {id}\nVENDOR: {p1}\nCLIENT: {p2}\nVALUE: USD {amt}\nGOVERNING LAW: India\n",
        b"LEASE_NO: {id}\nLESSOR: {p1}\nLESSEE: {p2}\nRENT: INR {amt}\nCITY: Bangalore\n",
    ]

    parties = [b"Alpha Corp", b"Beta Ltd", b"Gamma Inc", b"Delta Co", b"Epsilon LLC"]
    extensions = [
        b"\nAMENDMENT: Payment redirected to account XXXX-9999\n",
        b"\nADDENDUM: Jurisdiction changed to offshore territory\n",
        b"\nCLAUSE_17: All disputes waived by client\n",
        b"\nMODIFICATION: Penalty clause removed\n",
    ]

    samples = []

    # Legitimate contracts
    for i in range(n_legitimate):
        tmpl = random.choice(contract_templates)
        contract = tmpl.replace(
            b"{id}", f"EC-{i:05d}".encode()
        ).replace(b"{p1}", random.choice(parties)
        ).replace(b"{p2}", random.choice(parties)
        ).replace(b"{amt}", f"{random.randint(10000,9999999):,}".encode())

        sig = hashlib.sha256(secret_key + contract).hexdigest()
        feat = extract_features(contract, sig, label=0)
        samples.append(features_to_vector(feat))

    # Forged contracts (length-extended)
    for i in range(n_forged):
        tmpl = random.choice(contract_templates)
        base_contract = tmpl.replace(
            b"{id}", f"EC-{i:05d}".encode()
        ).replace(b"{p1}", random.choice(parties)
        ).replace(b"{p2}", random.choice(parties)
        ).replace(b"{amt}", f"{random.randint(10000,9999999):,}".encode())

        original_sig = hashlib.sha256(secret_key + base_contract).hexdigest()

        # Simulate length-extension: append MD padding + malicious clause
        ext = random.choice(extensions)
        padding = md_padding(len(secret_key) + len(base_contract))
        forged_contract = base_contract + padding + ext

        # Forged signature has same hex entropy as real SHA-256 output
        forged_sig = hashlib.sha256(os.urandom(32)).hexdigest()  # placeholder

        feat = extract_features(forged_contract, forged_sig, label=1)
        samples.append(features_to_vector(feat))

    X = np.array(samples)
    y = np.array([0]*n_legitimate + [1]*n_forged, dtype=int)
    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


# ─── Threshold-Based Anomaly Detector ────────────────────────────────────────

class StaticThresholdDetector:
    """
    Baseline static threshold detector (Category B1 in taxonomy).
    Flags contracts whose payload length or null-byte ratio exceeds fixed thresholds.
    Simple, fast, but poorly adapted to natural contract length variability.
    """

    def __init__(self, max_length: int = 800, max_null_ratio: float = 0.12):
        self.max_length   = max_length
        self.max_null_ratio = max_null_ratio

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X columns: payload_length=0, null_byte_ratio=3
        length_flag    = X[:, 0] > self.max_length
        null_flag      = X[:, 3] > self.max_null_ratio
        return (length_flag | null_flag).astype(int)


class EntropyAnomalyDetector:
    """
    Entropy-based dynamic detector (Category B2c in taxonomy).
    Length-extended contracts exhibit low-entropy padding regions
    due to MD structural 0x00 padding bytes.
    """

    def __init__(self):
        self.threshold_entropy       = None   # Minimum legitimate padding entropy
        self.threshold_null_ratio    = None   # Maximum legitimate null ratio
        self.threshold_high_ratio    = None   # Maximum legitimate high-byte ratio

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Learn thresholds from legitimate training samples."""
        legit = X_train[y_train == 0]
        # Set thresholds at mean - 2*std for entropy, mean + 2*std for ratios
        self.threshold_entropy    = legit[:, 2].mean() - 2 * legit[:, 2].std()
        self.threshold_null_ratio = legit[:, 3].mean() + 2 * legit[:, 3].std()
        self.threshold_high_ratio = legit[:, 4].mean() + 2 * legit[:, 4].std()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        low_entropy  = X[:, 2] < self.threshold_entropy
        high_null    = X[:, 3] > self.threshold_null_ratio
        high_highbyt = X[:, 4] > self.threshold_high_ratio
        return (low_entropy | high_null | high_highbyt).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict(X)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        dr  = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return {"DR": round(dr*100, 2), "FPR": round(fpr*100, 2),
                "TP": tp, "FP": fp, "TN": tn, "FN": fn}


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    dr  = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    acc = (tp + tn) / len(y_true)
    print(f"  {name:<35}  DR={dr*100:5.1f}%  FPR={fpr*100:5.1f}%  Acc={acc*100:5.1f}%")
    return {"DR": dr, "FPR": fpr, "Acc": acc}


def main():
    print("=" * 70)
    print("  Hash Integrity Anomaly Detector")
    print("  Paper: Cryptographic Forgery in E-Contracts")
    print("=" * 70)

    # Generate dataset
    print("\n[1] Generating dataset (500 legitimate + 500 forged)...")
    X, y = generate_dataset(n_legitimate=500, n_forged=500)

    # Train/test split (80/20)
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"    Train: {len(y_train)} samples | Test: {len(y_test)} samples")

    print("\n[2] Evaluating detectors on test set:")
    print("-" * 70)

    # Static threshold
    static_det = StaticThresholdDetector()
    evaluate("Static Threshold Detector", y_test, static_det.predict(X_test))

    # Entropy-based
    entropy_det = EntropyAnomalyDetector()
    entropy_det.fit(X_train, y_train)
    evaluate("Entropy Anomaly Detector", y_test, entropy_det.predict(X_test))

    print("\n[3] Feature statistics (legitimate vs forged):")
    print("-" * 70)
    feature_names = [
        "payload_length", "entropy", "padding_entropy",
        "null_byte_ratio", "high_byte_ratio", "block_boundary",
        "length_mod_64", "hash_entropy"
    ]
    legit_X = X[y == 0]
    forged_X = X[y == 1]
    print(f"  {'Feature':<25}  {'Legit Mean':>12}  {'Forged Mean':>12}  {'Δ':>10}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*10}")
    for i, name in enumerate(feature_names):
        lm = legit_X[:, i].mean()
        fm = forged_X[:, i].mean()
        print(f"  {name:<25}  {lm:>12.4f}  {fm:>12.4f}  {fm-lm:>+10.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
