"""
core/crypto_engine.py
=====================
Cryptographic Forgery in E-Contracts
Core Engine: Length-Extension Attack + HMAC Defense

Implements:
  - Merkle-Damgård SHA-256 padding
  - SHA-256 internal state manipulation (length-extension)
  - HMAC-SHA256 defense
  - Attack result packaging for API response
"""

import hashlib
import struct
import hmac as hmac_lib
import os
import time


# ── SHA-256 constants ─────────────────────────────────────────────────────────
SHA256_K = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
]

def _rotr32(x, n):
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def _sha256_compress(state, block):
    """One SHA-256 compression round on a 64-byte block from arbitrary state."""
    w = list(struct.unpack('>16I', block))
    for i in range(16, 64):
        s0 = _rotr32(w[i-15],7) ^ _rotr32(w[i-15],18) ^ (w[i-15]>>3)
        s1 = _rotr32(w[i-2],17) ^ _rotr32(w[i-2],19)  ^ (w[i-2]>>10)
        w.append((w[i-16]+s0+w[i-7]+s1) & 0xFFFFFFFF)
    a,b,c,d,e,f,g,h = state
    for i in range(64):
        S1    = _rotr32(e,6)^_rotr32(e,11)^_rotr32(e,25)
        ch    = (e&f)^((~e)&g)
        temp1 = (h+S1+ch+SHA256_K[i]+w[i]) & 0xFFFFFFFF
        S0    = _rotr32(a,2)^_rotr32(a,13)^_rotr32(a,22)
        maj   = (a&b)^(a&c)^(b&c)
        temp2 = (S0+maj) & 0xFFFFFFFF
        h,g,f,e = g,f,e,(d+temp1)&0xFFFFFFFF
        d,c,b,a = c,b,a,(temp1+temp2)&0xFFFFFFFF
    return [(state[i]+v)&0xFFFFFFFF for i,v in enumerate([a,b,c,d,e,f,g,h])]


def md_padding(message_length: int) -> bytes:
    """Merkle-Damgård padding for a message of given byte length."""
    bit_len = message_length * 8
    pad = b'\x80'
    pad_len = (56 - (message_length + 1) % 64) % 64
    pad += b'\x00' * pad_len
    pad += struct.pack('>Q', bit_len)
    return pad


def sha256_from_state(state_hex: str, extra_data: bytes, original_msg_len: int) -> str:
    """
    Continue SHA-256 from a known output state — core of length-extension.
    """
    state = list(struct.unpack('>8I', bytes.fromhex(state_hex)))
    padded_len = original_msg_len + len(md_padding(original_msg_len))
    ext_pad = extra_data + md_padding(padded_len + len(extra_data))
    for i in range(0, len(ext_pad), 64):
        block = ext_pad[i:i+64]
        if len(block) == 64:
            state = _sha256_compress(state, block)
    return ''.join(f'{x:08x}' for x in state)


# ── Attack & Defense ──────────────────────────────────────────────────────────

def vulnerable_sign(secret: bytes, contract: bytes) -> str:
    """INSECURE: raw SHA-256(secret || contract) — vulnerable to length-extension."""
    return hashlib.sha256(secret + contract).hexdigest()

def vulnerable_verify(secret: bytes, contract: bytes, signature: str) -> bool:
    return vulnerable_sign(secret, contract) == signature

def secure_sign(secret: bytes, contract: bytes) -> str:
    """SECURE: HMAC-SHA256(secret, contract) — immune to length-extension."""
    return hmac_lib.new(secret, contract, hashlib.sha256).hexdigest()

def secure_verify(secret: bytes, contract: bytes, signature: str) -> bool:
    expected = secure_sign(secret, contract)
    return hmac_lib.compare_digest(expected, signature)


def perform_attack(contract: bytes, signature: str, malicious: bytes,
                   secret_len: int) -> dict:
    """
    Execute hash length-extension attack.
    Returns full attack breakdown for UI display.
    """
    t0 = time.perf_counter()
    padding = md_padding(secret_len + len(contract))
    forged_contract = contract + padding + malicious
    forged_sig = sha256_from_state(signature, malicious, secret_len + len(contract))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "original_contract_hex":  contract.hex(),
        "original_contract_text": contract.decode('utf-8', errors='replace'),
        "original_length":        len(contract),
        "original_signature":     signature,
        "padding_hex":            padding.hex(),
        "padding_length":         len(padding),
        "padding_null_bytes":     padding.count(0x00),
        "malicious_hex":          malicious.hex(),
        "malicious_text":         malicious.decode('utf-8', errors='replace'),
        "malicious_length":       len(malicious),
        "forged_contract_hex":    forged_contract.hex(),
        "forged_contract_text":   forged_contract.decode('utf-8', errors='replace'),
        "forged_length":          len(forged_contract),
        "forged_signature":       forged_sig,
        "elapsed_ms":             round(elapsed_ms, 3),
        "signatures_differ":      signature != forged_sig,
    }
