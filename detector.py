import hashlib
import hmac
import struct
import numpy as np
import math
import random
import codecs
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# ─── ROTATION HELPERS ────────────────────────────────────────────────────────
def right_rotate(value, shift):
    return ((value >> shift) | (value << (32 - shift))) & 0xffffffff

def left_rotate(value, shift):
    return ((value << shift) | (value >> (32 - shift))) & 0xffffffff

# ─── ESCAPE CODES PARSER ─────────────────────────────────────────────────────
def decode_payload_to_bytes(msg: str) -> bytes:
    """Decodes a string that may contain hex escape sequences (e.g., \\x80) to raw bytes."""
    if '\\x' in msg:
        try:
            return codecs.escape_decode(msg.encode('utf-8'))[0]
        except Exception:
            pass
    return msg.encode('utf-8')

# ─── CUSTOM COMPRESSION ENGINES (FOR STATE RESTORATION) ──────────────────────

# SHA-256 Constants
K_256 = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

def sha256_compress(state, block):
    """SHA-256 block compression function."""
    w = [0] * 64
    for i in range(16):
        w[i] = struct.unpack('>I', block[i*4 : i*4+4])[0]
    for i in range(16, 64):
        s0 = (right_rotate(w[i-15], 7) ^ right_rotate(w[i-15], 18) ^ (w[i-15] >> 3)) & 0xffffffff
        s1 = (right_rotate(w[i-2], 17) ^ right_rotate(w[i-2], 19) ^ (w[i-2] >> 10)) & 0xffffffff
        w[i] = (w[i-16] + s0 + w[i-7] + s1) & 0xffffffff

    a, b, c, d, e, f, g, h = state

    for i in range(64):
        S1 = (right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25)) & 0xffffffff
        ch = (e & f) ^ ((~e) & g)
        temp1 = (h + S1 + ch + K_256[i] + w[i]) & 0xffffffff
        S0 = (right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22)) & 0xffffffff
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (S0 + maj) & 0xffffffff

        h = g
        g = f
        f = e
        e = (d + temp1) & 0xffffffff
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xffffffff

    return [
        (state[0] + a) & 0xffffffff,
        (state[1] + b) & 0xffffffff,
        (state[2] + c) & 0xffffffff,
        (state[3] + d) & 0xffffffff,
        (state[4] + e) & 0xffffffff,
        (state[5] + f) & 0xffffffff,
        (state[6] + g) & 0xffffffff,
        (state[7] + h) & 0xffffffff
    ]

def sha1_compress(state, block):
    """SHA-1 block compression function."""
    w = [0] * 80
    for i in range(16):
        w[i] = struct.unpack('>I', block[i*4 : i*4+4])[0]
    for i in range(16, 80):
        w[i] = left_rotate(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1)

    a, b, c, d, e = state

    for i in range(80):
        if 0 <= i <= 19:
            f = (b & c) | ((~b) & d)
            k = 0x5A827999
        elif 20 <= i <= 39:
            f = b ^ c ^ d
            k = 0x6ED9EBA1
        elif 40 <= i <= 59:
            f = (b & c) | (b & d) | (c & d)
            k = 0x8F1BBCDC
        elif 60 <= i <= 79:
            f = b ^ c ^ d
            k = 0xCA62C1D6

        temp = (left_rotate(a, 5) + f + e + k + w[i]) & 0xffffffff
        e = d
        d = c
        c = left_rotate(b, 30)
        b = a
        a = temp

    return [
        (state[0] + a) & 0xffffffff,
        (state[1] + b) & 0xffffffff,
        (state[2] + c) & 0xffffffff,
        (state[3] + d) & 0xffffffff,
        (state[4] + e) & 0xffffffff
    ]

# MD5 Constants
s_md5 = [
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
]
K_MD5 = [int(4294967296 * abs(math.sin(i + 1))) & 0xffffffff for i in range(64)]

def md5_compress(state, block):
    """MD5 block compression function."""
    M = list(struct.unpack('<16I', block))
    A, B, C, D = state
    a, b, c, d = A, B, C, D

    for i in range(64):
        if 0 <= i <= 15:
            f = (b & c) | ((~b) & d)
            g = i
        elif 16 <= i <= 31:
            f = (d & b) | ((~d) & c)
            g = (5 * i + 1) % 16
        elif 32 <= i <= 47:
            f = b ^ c ^ d
            g = (3 * i + 5) % 16
        elif 48 <= i <= 63:
            f = c ^ (b | (~d))
            g = (7 * i) % 16

        to_rotate = (a + f + K_MD5[i] + M[g]) & 0xffffffff
        rot = left_rotate(to_rotate, s_md5[i])
        temp = (b + rot) & 0xffffffff

        a = d
        d = c
        c = b
        b = temp

    return [
        (A + a) & 0xffffffff,
        (B + b) & 0xffffffff,
        (C + c) & 0xffffffff,
        (D + d) & 0xffffffff
    ]

# ─── MD PADDING CALCULATOR ───────────────────────────────────────────────────
def md_padding(message_length: int, algo: str = "sha256") -> bytes:
    """Compute Merkle–Damgård padding for a given message length in bytes."""
    padding = b'\x80'
    # Block size is 64 bytes. Leaving 8 bytes for length field.
    nulls_len = (55 - message_length) % 64
    padding += b'\x00' * nulls_len
    
    if algo == "md5":
        padding += struct.pack('<Q', message_length * 8)
    else:
        padding += struct.pack('>Q', message_length * 8)
        
    return padding

# ─── TRUE LENGTH EXTENSION ATTACK ENGINE ─────────────────────────────────────
def length_extension_attack(algo: str, original_tag: str, original_msg: str, extension: str, secret_len: int):
    """
    Performs an actual cryptographic length-extension attack.
    Reconstructs internal state from the original tag, pads the original message,
    and hashes the extension using the reconstructed state as the initial vector.
    Returns: (forged_tag, forged_display_string, forged_payload_hex)
    """
    original_msg_bytes = decode_payload_to_bytes(original_msg)
    extension_bytes = decode_payload_to_bytes(extension)
    
    # 1. Compute padding for the original message (including the secret key length)
    original_total_len = secret_len + len(original_msg_bytes)
    orig_padding = md_padding(original_total_len, algo)
    
    # The forged message bytes (what the server will see as input)
    forged_msg_bytes = original_msg_bytes + orig_padding + extension_bytes
    
    # 2. Reconstruct internal state registers from the original tag
    tag_bytes = bytes.fromhex(original_tag)
    
    if algo == "sha256":
        if len(tag_bytes) != 32:
            raise ValueError("SHA-256 tag must be 32 bytes (64 hex characters)")
        state = list(struct.unpack('>8I', tag_bytes))
    elif algo == "sha1":
        if len(tag_bytes) != 20:
            raise ValueError("SHA-1 tag must be 20 bytes (40 hex characters)")
        state = list(struct.unpack('>5I', tag_bytes))
    elif algo == "md5":
        if len(tag_bytes) != 16:
            raise ValueError("MD5 tag must be 16 bytes (32 hex characters)")
        state = list(struct.unpack('<4I', tag_bytes))
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
        
    # 3. Process the extension block by block starting from the reconstructed state.
    # The extension is appended to the padded original message.
    forged_total_len = original_total_len + len(orig_padding) + len(extension_bytes)
    ext_padding = md_padding(forged_total_len, algo)
    ext_padded_data = extension_bytes + ext_padding
    
    # Compress the padded extension data block-by-block
    blocks = [ext_padded_data[i : i + 64] for i in range(0, len(ext_padded_data), 64)]
    
    for block in blocks:
        if algo == "sha256":
            state = sha256_compress(state, block)
        elif algo == "sha1":
            state = sha1_compress(state, block)
        elif algo == "md5":
            state = md5_compress(state, block)
            
    # 5. Pack the final state to get the forged tag
    if algo == "sha256":
        forged_tag = struct.pack('>8I', *state).hex()
    elif algo == "sha1":
        forged_tag = struct.pack('>5I', *state).hex()
    elif algo == "md5":
        forged_tag = struct.pack('<4I', *state).hex()
        
    # Represent the forged message for display (converting binary padding to visible escapes)
    escaped_padding = "".join(f"\\x{b:02x}" for b in orig_padding)
    forged_display = original_msg + escaped_padding + extension
    
    return forged_tag, forged_display, forged_msg_bytes.hex()

# ─── VULNERABLE & SECURE SIGNATURE METHODS ───────────────────────────────────
def vulnerable_sign(secret: str, message: str, algo: str = "sha256") -> str:
    """Vulnerable: H(secret || message) — susceptible to length extension."""
    m_bytes = decode_payload_to_bytes(message)
    data = secret.encode('utf-8') + m_bytes
    if algo == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algo == "sha1":
        return hashlib.sha1(data).hexdigest()
    elif algo == "md5":
        return hashlib.md5(data).hexdigest()
    return hashlib.sha256(data).hexdigest()

def secure_sign(secret: str, message: str, algo: str = "sha256") -> str:
    """Secure: HMAC — not susceptible to length extension."""
    m_bytes = decode_payload_to_bytes(message)
    s_bytes = secret.encode('utf-8')
    if algo == "sha256":
        return hmac.new(s_bytes, m_bytes, hashlib.sha256).hexdigest()
    elif algo == "sha1":
        return hmac.new(s_bytes, m_bytes, hashlib.sha1).hexdigest()
    elif algo == "md5":
        return hmac.new(s_bytes, m_bytes, hashlib.md5).hexdigest()
    return hmac.new(s_bytes, m_bytes, hashlib.sha256).hexdigest()

# ─── SIMULATED ANNEALING FOR SECRET LENGTH SEARCH ────────────────────────────
def simulated_annealing_secret_length(hash_hex, max_len=64, T0=100, alpha=0.95, iterations=300):
    """
    Simulates an attacker using SA to guess the secret key length.
    Returns the guessed length and convergence log.
    """
    current = random.randint(1, max_len)
    T = T0
    log = []

    # Heuristic scoring: lengths divisible by block/word alignments are more probable
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

# ─── FEATURE EXTRACTION FOR MACHINE LEARNING ──────────────────────────────────
def extract_features(payload: str, tag: str):
    """
    Extract 8 features from a contract payload + tag.
    Handles hex-escaped strings, surrogates, and raw bytes.
    """
    pb = decode_payload_to_bytes(payload)
    length = len(pb)
    
    # Detect padding bytes presence (\x80 or literal escape text)
    has_padding_byte = 1 if (b'\x80' in pb or '\\x80' in payload) else 0
    block_alignment = length % 64
    tag_entropy = len(set(tag)) / len(tag) if tag else 0.0
    
    # Ratio of null bytes (\x00)
    null_bytes = pb.count(b'\x00')
    null_byte_ratio = null_bytes / max(length, 1)
    
    length_mod_512 = (length * 8) % 512
    avg_byte = sum(pb) / max(length, 1) if length > 0 else 0
    payload_entropy = len(set(pb)) / 256.0 if length > 0 else 0

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

# ─── DATASET GENERATION ──────────────────────────────────────────────────────
CONTRACT_TEMPLATES = [
    "Party A agrees to pay Party B $1000 for consulting services.",
    "This agreement is made between Client and Agency on July 19, 2026.",
    "The service provider will deliver the software package by next Monday.",
    "Non-disclosure agreement: both parties agree to keep code confidential.",
    "Rent agreement: tenant will pay $1500 monthly by the 5th of each month.",
    "The employee will receive a basic salary of $4500 and health insurance.",
    "Partnership agreement between TechCorp and SoftSystems.",
    "Vendor agrees to supply 50 units of server hardware by August.",
    "The contractor is responsible for all licensing fees and taxes.",
    "This contract may be terminated by either party with 30 days notice."
]

def generate_dataset_data(n_samples=600):
    X = []
    y = []
    random.seed(42)
    
    for i in range(n_samples):
        # Legitimate
        base = random.choice(CONTRACT_TEMPLATES)
        random_padding_text = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789 ") for _ in range(random.randint(0, 100)))
        payload = base + " " + random_padding_text
        tag = hashlib.sha256(payload.encode()).hexdigest()
        X.append(extract_features(payload, tag))
        y.append(0)
        
        # Forged (simulate MD padding injection)
        base_forge = random.choice(CONTRACT_TEMPLATES)
        secret_len = random.randint(8, 32)
        total_len = secret_len + len(base_forge.encode())
        
        padding = md_padding(total_len, "sha256")
        extension = " | FORGED CLAUSE: Transfer $" + str(random.randint(1000, 99999)) + " to account " + str(random.randint(100, 999))
        
        # Payload showing typical signature of length extension tampering
        forged_payload = base_forge + padding.decode('utf-8', errors='ignore') + extension
        forged_tag = hashlib.sha256((base_forge + extension).encode()).hexdigest()
        X.append(extract_features(forged_payload, forged_tag))
        y.append(1)
        
    return np.array(X), np.array(y)

# ─── INTERACTIVE ML CLASSIFIER STUDIO ────────────────────────────────────────
ACTIVE_MODEL = None

def train_custom_classifier(model_type='rf', hyperparams=None):
    if hyperparams is None:
        hyperparams = {}
    
    X, y = generate_dataset_data(n_samples=600)
    test_split = float(hyperparams.get('test_split', 0.2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    
    if model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=int(hyperparams.get('n_estimators', 100)),
            max_depth=hyperparams.get('max_depth', None) if hyperparams.get('max_depth') else None,
            random_state=42
        )
    elif model_type == 'dt':
        clf = DecisionTreeClassifier(
            max_depth=hyperparams.get('max_depth', None) if hyperparams.get('max_depth') else None,
            random_state=42
        )
    elif model_type == 'lr':
        clf = LogisticRegression(
            C=float(hyperparams.get('C', 1.0)),
            max_iter=1000,
            random_state=42
        )
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # ROC Curve Points
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    # Downsample points for cleaner transmission/plotting
    step = max(1, len(fpr) // 20)
    roc_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in range(0, len(fpr), step)]
    if len(roc_points) == 0 or roc_points[-1]["fpr"] != 1.0 or roc_points[-1]["tpr"] != 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})
        
    roc_auc = auc(fpr, tpr)
    
    # Feature Importances
    feature_names = [
        "payload_length",
        "padding_byte_detected",
        "block_alignment",
        "tag_entropy",
        "null_byte_ratio",
        "length_mod_512",
        "average_byte_value",
        "payload_entropy"
    ]
    
    importances = {}
    if hasattr(clf, 'feature_importances_'):
        for name, imp in zip(feature_names, clf.feature_importances_):
            importances[name] = round(float(imp), 4)
    elif hasattr(clf, 'coef_'):
        coefs = np.abs(clf.coef_[0])
        total = sum(coefs) if sum(coefs) > 0 else 1
        for name, imp in zip(feature_names, coefs / total):
            importances[name] = round(float(imp), 4)
    else:
        for name in feature_names:
            importances[name] = 0.125
            
    # Keep trained model active
    global ACTIVE_MODEL
    ACTIVE_MODEL = clf
    
    return {
        "model_type": model_type,
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "auc": round(float(roc_auc), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "roc_curve": roc_points,
        "feature_importances": importances
    }

def detect_forgery(payload: str, tag: str):
    """Runs the trained classifier on an incoming payload + tag."""
    features = extract_features(payload, tag)
    feat_arr = np.array(features).reshape(1, -1)
    
    if ACTIVE_MODEL is None:
        train_custom_classifier('rf')
        
    pred = ACTIVE_MODEL.predict(feat_arr)[0]
    proba = ACTIVE_MODEL.predict_proba(feat_arr)[0] if hasattr(ACTIVE_MODEL, "predict_proba") else [1.0 - pred, pred]
    
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

# Initialize model
train_custom_classifier('rf')