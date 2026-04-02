import hashlib
import hmac
import struct
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ─── Simulated Annealing for secret length search ────────────────────────────
import math, random

def simulated_annealing_secret_length(hash_hex, max_len=64, T0=100, alpha=0.95, iterations=300):
    """
    Simulates an attacker using SA to guess the secret key length.
    Returns the guessed length and convergence log.
    """
    current = random.randint(1, max_len)
    T = T0
    log = []

    # Fake scoring: prefer lengths divisible by block alignment (realistic heuristic)
    def score(l):
        padded_total = l + 64  # assume 64-byte message
        blocks = math.ceil((padded_total + 9) / 64)
        boundary_score = 1.0 / (1 + abs((blocks * 64) - (padded_total + 9)))
        alignment_bonus = 1.0 if l % 8 == 0 else 0.5
        return boundary_score + alignment_bonus * 0.1

    best = current
    best_score = score(current)

    for i in range(iterations):
        neighbor = current + random.choice([-2, -1, 1, 2])
        neighbor = max(1, min(max_len, neighbor))
        delta = score(neighbor) - score(current)
        if delta > 0 or random.random() < math.exp(delta / T):
            current = neighbor
        if score(current) > best_score:
            best = current
            best_score = score(current)
        T *= alpha
        if i % 30 == 0:
            log.append({"iteration": i, "temperature": round(T, 4), "current_guess": current})

    return best, log


# ─── Vulnerable MAC (raw SHA-256) ────────────────────────────────────────────
def vulnerable_sign(secret: str, message: str) -> str:
    """Vulnerable: H(secret || message) — susceptible to length extension."""
    data = (secret + message).encode()
    return hashlib.sha256(data).hexdigest()


# ─── Secure MAC (HMAC-SHA256) ─────────────────────────────────────────────────
def secure_sign(secret: str, message: str) -> str:
    """Secure: HMAC-SHA256 — not susceptible to length extension."""
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


# ─── MD Padding Calculator ────────────────────────────────────────────────────
def md_padding(message_length: int) -> bytes:
    """Compute Merkle–Damgård padding for a given message length."""
    padding = b'\x80'
    padding += b'\x00' * ((55 - message_length) % 64)
    padding += struct.pack('<Q', message_length * 8)
    return padding


# ─── Length Extension Attack Simulator ───────────────────────────────────────
def length_extension_attack(original_tag: str, original_msg: str, extension: str, secret_len: int):
    """
    Simulates a hash length-extension attack.
    Given a known tag H(secret||msg), forge a new tag for (msg||padding||extension).
    Returns forged tag and the full forged message representation.
    """
    original_len = secret_len + len(original_msg.encode())
    padding = md_padding(original_len)

    # Reconstruct internal state from tag
    tag_bytes = bytes.fromhex(original_tag)
    state = struct.unpack('>8I', tag_bytes)

    # Continue hashing from that state
    h = hashlib.sha256()
    # Inject internal state (simulate resuming)
    # For demo: we re-hash with the state seed to simulate extension
    forged_input = original_msg.encode() + padding + extension.encode()
    resumed = hashlib.sha256(tag_bytes + extension.encode()).hexdigest()

    forged_display = original_msg + "\\x80[PADDING]" + extension
    return resumed, forged_display


# ─── Feature Extraction for ML ───────────────────────────────────────────────
def extract_features(payload: str, tag: str):
    """
    Extract 8 features from a contract payload + tag for the Random Forest.
    """
    pb = payload.encode()
    length = len(pb)
    has_padding_byte = 1 if b'\\x80' in pb or 0x80 in pb else 0
    block_alignment = length % 64
    tag_entropy = len(set(tag)) / 16.0
    null_byte_ratio = pb.count(b'\x00') / max(length, 1)
    length_mod_512 = length % 512
    avg_byte = sum(pb) / max(length, 1)
    payload_entropy = len(set(pb)) / 256.0

    return [
        length,
        has_padding_byte,
        block_alignment,
        tag_entropy,
        null_byte_ratio,
        length_mod_512,
        avg_byte,
        payload_entropy
    ]


# ─── Train Random Forest (synthetic data) ────────────────────────────────────
def train_detector():
    """
    Train a Random Forest classifier on synthetic legitimate vs forged samples.
    Returns fitted model.
    """
    np.random.seed(42)
    n = 600

    # Legitimate samples
    legit_lengths     = np.random.randint(20, 300, n)
    legit_padding     = np.zeros(n)
    legit_alignment   = legit_lengths % 64
    legit_entropy     = np.random.uniform(0.7, 1.0, n)
    legit_null        = np.zeros(n)
    legit_mod512      = legit_lengths % 512
    legit_avg         = np.random.uniform(60, 122, n)
    legit_pay_entropy = np.random.uniform(0.5, 0.9, n)

    # Forged samples (length-extension artifacts)
    forge_lengths     = np.random.randint(80, 450, n)
    forge_padding     = np.ones(n)                        # padding byte present
    forge_alignment   = np.zeros(n)                       # perfect block alignment
    forge_entropy     = np.random.uniform(0.4, 0.75, n)
    forge_null        = np.random.uniform(0.05, 0.3, n)   # null bytes from padding
    forge_mod512      = np.zeros(n)
    forge_avg         = np.random.uniform(20, 80, n)
    forge_pay_entropy = np.random.uniform(0.2, 0.6, n)

    X_legit = np.column_stack([legit_lengths, legit_padding, legit_alignment,
                                legit_entropy, legit_null, legit_mod512,
                                legit_avg, legit_pay_entropy])
    X_forge = np.column_stack([forge_lengths, forge_padding, forge_alignment,
                                forge_entropy, forge_null, forge_mod512,
                                forge_avg, forge_pay_entropy])

    X = np.vstack([X_legit, X_forge])
    y = np.array([0] * n + [1] * n)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf


# ─── Global model (trained once on import) ───────────────────────────────────
MODEL = train_detector()


def detect_forgery(payload: str, tag: str):
    """
    Run the Random Forest detector on a payload+tag pair.
    Returns dict with prediction, confidence, and feature breakdown.
    """
    features = extract_features(payload, tag)
    feat_arr = np.array(features).reshape(1, -1)
    pred = MODEL.predict(feat_arr)[0]
    proba = MODEL.predict_proba(feat_arr)[0]

    return {
        "is_forged": bool(pred),
        "confidence": round(float(max(proba)) * 100, 2),
        "forgery_probability": round(float(proba[1]) * 100, 2),
        "legitimate_probability": round(float(proba[0]) * 100, 2),
        "features": {
            "payload_length": features[0],
            "padding_byte_detected": bool(features[1]),
            "block_alignment": features[2],
            "tag_entropy": round(features[3], 4),
            "null_byte_ratio": round(features[4], 4),
            "length_mod_512": features[5],
            "average_byte_value": round(features[6], 2),
            "payload_entropy": round(features[7], 4)
        }
    }