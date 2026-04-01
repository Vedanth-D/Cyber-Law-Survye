"""
=============================================================================
Cryptographic Forgery in E-Contracts: A Survey of Metaheuristic
Length-Extension Attacks
-----------------------------------------------------------------------------
Module 1: Hash Length-Extension Attack Simulation
=============================================================================

Demonstrates the structural vulnerability of Merkle-Damgård hash functions
(MD5, SHA-1, SHA-256) to length-extension attacks in e-contract contexts.

References:
  [3]  Kelsey & Schneier, "Second preimages on n-bit hash functions," EUROCRYPT 2005
  [19] NIST FIPS PUB 180-4, Secure Hash Standard
"""

import hashlib
import struct
import hmac
import os
from typing import Tuple

# ─── Merkle-Damgård Padding ───────────────────────────────────────────────────

def md_padding(message_length: int, block_size: int = 64) -> bytes:
    """
    Compute the Merkle-Damgård padding for a message of given byte length.
    Appends 0x80, zero bytes, then 64-bit big-endian message bit-length.

    Args:
        message_length: Length of the original message in bytes.
        block_size:     Hash block size in bytes (64 for MD5/SHA-1/SHA-256).

    Returns:
        Padding bytes to append to the message.
    """
    bit_length = message_length * 8
    padding = b'\x80'
    # Pad until length ≡ 56 (mod 64)
    pad_len = (56 - (message_length + 1) % block_size) % block_size
    padding += b'\x00' * pad_len
    # Append original bit length as 64-bit big-endian
    padding += struct.pack('>Q', bit_length)
    return padding


# ─── SHA-256 Internal State Manipulation ──────────────────────────────────────

# SHA-256 round constants
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

def rotr32(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def sha256_compress(state: list, block: bytes) -> list:
    """
    Perform one SHA-256 compression round on a 64-byte block.
    Allows initialization from an arbitrary internal state (needed for extension attack).

    Args:
        state: List of 8 x 32-bit integers representing current hash state.
        block: 64-byte message block.

    Returns:
        Updated 8-element state list.
    """
    assert len(block) == 64
    w = list(struct.unpack('>16I', block))
    for i in range(16, 64):
        s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)

    a, b, c, d, e, f, g, h = state
    for i in range(64):
        S1   = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)
        ch   = (e & f) ^ ((~e) & g)
        temp1 = (h + S1 + ch + K[i] + w[i]) & 0xFFFFFFFF
        S0   = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)
        maj  = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (S0 + maj) & 0xFFFFFFFF

        h, g, f, e = g, f, e, (d + temp1) & 0xFFFFFFFF
        d, c, b, a = c, b, a, (temp1 + temp2) & 0xFFFFFFFF

    return [
        (state[0] + a) & 0xFFFFFFFF, (state[1] + b) & 0xFFFFFFFF,
        (state[2] + c) & 0xFFFFFFFF, (state[3] + d) & 0xFFFFFFFF,
        (state[4] + e) & 0xFFFFFFFF, (state[5] + f) & 0xFFFFFFFF,
        (state[6] + g) & 0xFFFFFFFF, (state[7] + h) & 0xFFFFFFFF,
    ]


def sha256_from_state(state_hex: str, extra_data: bytes,
                       original_message_len: int) -> str:
    """
    Continue a SHA-256 computation from a known intermediate state.
    This is the core primitive of the length-extension attack.

    Args:
        state_hex:            Hex-encoded SHA-256 output to use as initial state.
        extra_data:           Additional bytes to hash (the forged extension).
        original_message_len: Byte length of the message that produced state_hex
                              (including secret key length).

    Returns:
        Hex string of forged SHA-256 hash.
    """
    # Reconstruct 8-word internal state from hash output
    state = list(struct.unpack('>8I', bytes.fromhex(state_hex)))

    # Account for padding of original message before starting extension
    padded_len = original_message_len + len(md_padding(original_message_len))
    message_len_so_far = padded_len

    # Process extension blocks
    data = extra_data
    # Pad extension data to block boundary
    pad = md_padding(message_len_so_far + len(extra_data))
    data_padded = extra_data + pad

    for i in range(0, len(data_padded), 64):
        block = data_padded[i:i+64]
        if len(block) < 64:
            break
        state = sha256_compress(state, block)
        message_len_so_far += 64

    return ''.join(f'{x:08x}' for x in state)


# ─── E-Contract Forgery Simulation ───────────────────────────────────────────

class EContractForger:
    """
    Simulates a length-extension forgery attack against an e-contract
    platform using raw SHA-256(secret || contract_bytes) authentication.

    In a real deployment this represents the attacker's position:
      - Can observe (contract_bytes, hash_value) for a signed contract
      - Knows the secret key length (or can enumerate it)
      - Wants to append malicious_clauses to produce a valid forgery
    """

    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key

    def sign_contract(self, contract: bytes) -> str:
        """Vulnerable signing: raw SHA-256(secret || contract). DO NOT USE IN PRODUCTION."""
        return hashlib.sha256(self.secret_key + contract).hexdigest()

    def verify_contract(self, contract: bytes, signature: str) -> bool:
        """Vulnerable verification against raw hash."""
        return self.sign_contract(contract) == signature

    def perform_extension_attack(
        self,
        observed_contract: bytes,
        observed_signature: str,
        malicious_extension: bytes,
        secret_key_length: int
    ) -> Tuple[bytes, str]:
        """
        Execute hash length-extension attack.

        Args:
            observed_contract:    Original contract bytes (known to attacker).
            observed_signature:   Valid SHA-256 signature of (secret || contract).
            malicious_extension:  Attacker-controlled clause to append.
            secret_key_length:    Known or guessed length of the secret key.

        Returns:
            (forged_contract, forged_signature) that will pass verification.
        """
        # Total length of (secret || contract) for padding calculation
        original_msg_len = secret_key_length + len(observed_contract)

        # The forged contract = original_contract + md_padding + malicious_extension
        padding = md_padding(original_msg_len)
        forged_contract = observed_contract + padding + malicious_extension

        # Compute forged signature from known hash state
        forged_sig = sha256_from_state(
            observed_signature,
            malicious_extension,
            original_msg_len
        )

        return forged_contract, forged_sig


# ─── HMAC Defense Demonstration ──────────────────────────────────────────────

class SecureEContractSigner:
    """
    Secure e-contract signing using HMAC-SHA256.
    HMAC is immune to length-extension attacks by construction.
    See: Bellare, Canetti, Krawczyk [7]
    """

    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key

    def sign_contract(self, contract: bytes) -> str:
        """Secure signing: HMAC-SHA256(secret, contract)."""
        return hmac.new(self.secret_key, contract, hashlib.sha256).hexdigest()

    def verify_contract(self, contract: bytes, signature: str) -> bool:
        """Constant-time HMAC verification."""
        expected = self.sign_contract(contract)
        return hmac.compare_digest(expected, signature)


# ─── Main Demo ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  E-Contract Length-Extension Attack Demonstration")
    print("  Paper: Cryptographic Forgery in E-Contracts")
    print("=" * 70)

    SECRET_KEY = b"JAIN_SECRET_2024"

    # ── Scenario: Vulnerable Platform ──
    print("\n[1] VULNERABLE PLATFORM (raw SHA-256)")
    print("-" * 50)

    original_contract = (
        b"CONTRACT_ID: EC-2024-0042\n"
        b"PARTIES: Alpha Corp, Beta Ltd\n"
        b"PAYMENT: INR 10,00,000\n"
        b"JURISDICTION: Bangalore, India\n"
    )

    malicious_clause = (
        b"\nAMENDMENT: Payment redirected to account XXXX-9999\n"
        b"JURISDICTION: Offshore, Tax Haven\n"
    )

    forger_system = EContractForger(SECRET_KEY)
    valid_signature = forger_system.sign_contract(original_contract)
    print(f"  Original contract bytes : {len(original_contract)}")
    print(f"  Valid signature         : {valid_signature[:32]}...")

    # Attacker performs length-extension
    forged_contract, forged_sig = forger_system.perform_extension_attack(
        observed_contract=original_contract,
        observed_signature=valid_signature,
        malicious_extension=malicious_clause,
        secret_key_length=len(SECRET_KEY)
    )

    # Verify forged contract passes the vulnerable verifier
    passes = forger_system.verify_contract(forged_contract, forged_sig)
    print(f"\n  [ATTACK] Forged contract bytes  : {len(forged_contract)}")
    print(f"  [ATTACK] Forged signature       : {forged_sig[:32]}...")
    print(f"  [ATTACK] Passes verification    : {passes}")
    print(f"\n  Malicious clause appended:")
    print(f"  {malicious_clause.decode(errors='replace')}")

    # ── Scenario: Secure Platform ──
    print("\n[2] SECURE PLATFORM (HMAC-SHA256)")
    print("-" * 50)

    secure_system = SecureEContractSigner(SECRET_KEY)
    secure_sig = secure_system.sign_contract(original_contract)
    print(f"  HMAC signature          : {secure_sig[:32]}...")

    # Attacker tries the same forged contract against secure verifier
    secure_passes = secure_system.verify_contract(forged_contract, forged_sig)
    print(f"  [ATTACK] Forged contract passes HMAC verification: {secure_passes}")
    print("\n  [RESULT] HMAC construction defeats length-extension attack.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
