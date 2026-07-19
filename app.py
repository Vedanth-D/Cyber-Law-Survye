from flask import Flask, request, jsonify, render_template, make_response
import hashlib
import time
from detector import (
    vulnerable_sign, secure_sign,
    length_extension_attack, detect_forgery,
    simulated_annealing_secret_length, md_padding,
    train_custom_classifier
)

app = Flask(__name__)
SECRET_KEY = "mysecretkey2024"

# ─── IMMUTABLE BLOCKCHAIN LEDGER SIMULATOR ──────────────────────────────────
class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        # Create genesis block
        self.create_block(previous_hash="0000000000000000000000000000000000000000000000000000000000000000", nonce=42)
        
    def create_block(self, previous_hash, nonce):
        merkle_root = self.calculate_merkle_root(self.pending_transactions)
        block = {
            "index": len(self.chain) + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "transactions": list(self.pending_transactions),
            "previous_hash": previous_hash,
            "merkle_root": merkle_root,
            "nonce": nonce,
            "hash": ""
        }
        block["hash"] = self.hash_block(block)
        self.pending_transactions = []
        self.chain.append(block)
        return block
        
    def hash_block(self, block):
        block_string = f"{block['index']}-{block['timestamp']}-{block['previous_hash']}-{block['merkle_root']}-{block['nonce']}"
        return hashlib.sha256(block_string.encode()).hexdigest()
        
    def calculate_merkle_root(self, txs):
        if not txs:
            return "0" * 64
        hashes = [hashlib.sha256(tx.encode()).hexdigest() for tx in txs]
        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    new_hashes.append(hashlib.sha256((hashes[i] + hashes[i+1]).encode()).hexdigest())
                else:
                    new_hashes.append(hashlib.sha256((hashes[i] + hashes[i]).encode()).hexdigest())
            hashes = new_hashes
        return hashes[0]
        
    def mine_block(self, difficulty=4):
        target = "0" * difficulty
        nonce = 0
        prev_hash = self.chain[-1]["hash"] if self.chain else "0" * 64
        merkle_root = self.calculate_merkle_root(self.pending_transactions)
        
        start_time = time.time()
        while True:
            block = {
                "index": len(self.chain) + 1,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "transactions": list(self.pending_transactions),
                "previous_hash": prev_hash,
                "merkle_root": merkle_root,
                "nonce": nonce
            }
            h = self.hash_block(block)
            if h.startswith(target):
                block["hash"] = h
                self.chain.append(block)
                self.pending_transactions = []
                duration = time.time() - start_time
                return block, duration
            nonce += 1

blockchain = Blockchain()

# ─── CORS HEADERS MIDDLEWARE ────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        resp = make_response()
        resp.headers["Access-Control-Allow-Origin"]  = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

# ─── WEB SERVER ROUTES ──────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/sign/vulnerable", methods=["GET", "POST", "OPTIONS"])
def sign_vulnerable():
    if request.method in ("GET", "OPTIONS"):
        return jsonify({"status": "endpoint ready"})
    data    = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    algo    = data.get("algo", "sha256").strip().lower()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    tag = vulnerable_sign(SECRET_KEY, message, algo)
    return jsonify({
        "method":  f"Raw {algo.upper()} (Vulnerable)",
        "message": message,
        "algo":    algo,
        "tag":     tag,
        "warning": f"This MAC is vulnerable to {algo.upper()} length-extension attacks!"
    })

@app.route("/api/sign/secure", methods=["GET", "POST", "OPTIONS"])
def sign_secure():
    if request.method in ("GET", "OPTIONS"):
        return jsonify({"status": "endpoint ready"})
    data    = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    algo    = data.get("algo", "sha256").strip().lower()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    tag = secure_sign(SECRET_KEY, message, algo)
    return jsonify({
        "method":  f"HMAC-{algo.upper()} (Secure)",
        "message": message,
        "algo":    algo,
        "tag":     tag,
        "info":    "HMAC double-hashing blocks Merkle-Damgård length extension."
    })

@app.route("/api/attack/length-extension", methods=["GET", "POST", "OPTIONS"])
def attack():
    if request.method in ("GET", "OPTIONS"):
        return jsonify({"status": "endpoint ready"})
    data         = request.get_json(force=True, silent=True) or {}
    original_tag = data.get("tag", "").strip()
    original_msg = data.get("message", "").strip()
    extension    = data.get("extension", " | FORGED CLAUSE: Pay attacker $99999")
    algo         = data.get("algo", "sha256").strip().lower()
    
    if not original_tag or not original_msg:
        return jsonify({"error": "tag and message are required"}), 400
        
    # Metaheuristic Search: guess secret length using simulated annealing
    guessed_len, sa_log = simulated_annealing_secret_length(original_tag)
    
    # Use user-provided secret length if present, else fallback to SA guess
    user_secret_len = data.get("secret_len")
    attack_len = int(user_secret_len) if (user_secret_len is not None and str(user_secret_len).isdigit()) else guessed_len
    
    # Real cryptographic hashing attack using the custom engine
    try:
        forged_tag, forged_msg_display, forged_payload_hex = length_extension_attack(
            algo, original_tag, original_msg, extension, attack_len
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
    padding_hex = md_padding(attack_len + len(original_msg.encode('utf-8')), algo).hex()
    
    return jsonify({
        "attack":                   f"{algo.upper()} Length-Extension Attack",
        "algo":                     algo,
        "original_tag":             original_tag,
        "original_message":         original_msg,
        "extension_appended":       extension,
        "sa_guessed_secret_length": guessed_len,
        "secret_length_used":       attack_len,
        "md_padding_hex":           padding_hex[:64] + ("..." if len(padding_hex) > 64 else ""),
        "forged_tag":               forged_tag,
        "forged_message_display":   forged_msg_display,
        "forged_payload_hex":       forged_payload_hex,
        "sa_convergence_log":       sa_log,
        "result":                   "Forged signature crafted successfully without knowing secret key!"
    })

@app.route("/api/detect", methods=["GET", "POST", "OPTIONS"])
def detect():
    if request.method in ("GET", "OPTIONS"):
        return jsonify({"status": "endpoint ready"})
    data    = request.get_json(force=True, silent=True) or {}
    payload = data.get("payload", "").strip()
    tag     = data.get("tag", "").strip()
    if not payload or not tag:
        return jsonify({"error": "payload and tag are required"}), 400
    result = detect_forgery(payload, tag)
    result["payload_preview"] = payload[:80] + ("..." if len(payload) > 80 else "")
    return jsonify(result)

@app.route("/api/verify", methods=["GET", "POST", "OPTIONS"])
def verify():
    if request.method in ("GET", "OPTIONS"):
        return jsonify({"status": "endpoint ready"})
    data    = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    tag     = data.get("tag", "").strip()
    method  = data.get("method", "secure").strip().lower()
    algo    = data.get("algo", "sha256").strip().lower()
    
    if not message or not tag:
        return jsonify({"error": "message and tag are required"}), 400
        
    # Check 1: Cryptographic signature verification
    expected = vulnerable_sign(SECRET_KEY, message, algo) if method == "vulnerable" \
               else secure_sign(SECRET_KEY, message, algo)
    crypto_valid = (expected == tag)
    
    # Check 2: ML Forgery detection check
    ml_result = detect_forgery(message, tag)
    ml_forgery = ml_result["is_forged"]
    
    # Check 3: Blockchain ledger anchor check
    blockchain_anchored = False
    for block in blockchain.chain:
        if tag in block["transactions"]:
            blockchain_anchored = True
            break
            
    # Compile Verdicts
    if crypto_valid:
        if ml_forgery:
            verdict = "TAMPERED CONTRACT (Length-Extension Signature Match but ML Flagged)"
        elif method == "vulnerable" and not blockchain_anchored:
            verdict = "SUSPICIOUS CONTRACT (Signature matches, but not anchored to ledger)"
        else:
            verdict = "VALID CONTRACT"
    else:
        verdict = "INVALID SIGNATURE (Tampered or incorrect key)"
        
    return jsonify({
        "crypto_valid":         crypto_valid,
        "ml_forged":            ml_forgery,
        "blockchain_anchored":  blockchain_anchored,
        "method":        method,
        "algo":          algo,
        "submitted_tag": tag,
        "expected_tag":  expected,
        "verdict":       verdict
    })

# ─── MACHINE LEARNING TRAINING API ──────────────────────────────────────────
@app.route("/api/ml/train", methods=["POST", "OPTIONS"])
def ml_train():
    data        = request.get_json(force=True, silent=True) or {}
    model_type  = data.get("model_type", "rf").strip().lower()
    hyperparams = data.get("hyperparams", {})
    
    try:
        report = train_custom_classifier(model_type, hyperparams)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─── BLOCKCHAIN LEDGER APIS ─────────────────────────────────────────────────
@app.route("/api/blockchain/anchor", methods=["POST", "OPTIONS"])
def blockchain_anchor():
    data = request.get_json(force=True, silent=True) or {}
    tag  = data.get("tag", "").strip()
    if not tag:
        return jsonify({"error": "No transaction hash provided"}), 400
        
    # Check if transaction is already anchored or pending
    if tag in blockchain.pending_transactions:
        return jsonify({"info": "Transaction already pending in block", "status": "pending"})
        
    for block in blockchain.chain:
        if tag in block["transactions"]:
            return jsonify({"info": f"Transaction already mined in block #{block['index']}", "status": "mined"})
            
    blockchain.pending_transactions.append(tag)
    return jsonify({
        "status": "success",
        "pending_transactions_count": len(blockchain.pending_transactions),
        "message": "Transaction hash successfully added to pending block!"
    })

@app.route("/api/blockchain/mine", methods=["POST", "OPTIONS"])
def blockchain_mine():
    data       = request.get_json(force=True, silent=True) or {}
    difficulty = int(data.get("difficulty", 4))
    
    if not blockchain.pending_transactions:
        return jsonify({"error": "No pending transactions to mine!"}), 400
        
    block, duration = blockchain.mine_block(difficulty)
    return jsonify({
        "status": "success",
        "block": block,
        "mining_time_seconds": round(duration, 4),
        "message": f"Block #{block['index']} mined successfully after searching nonces!"
    })

@app.route("/api/blockchain/blocks", methods=["GET"])
def blockchain_blocks():
    return jsonify({
        "chain": blockchain.chain,
        "pending_transactions": blockchain.pending_transactions
    })

@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "model": "Random Forest (Dynamic Studio)",
        "training_samples": 1200,
        "features": 8,
        "hash_functions": ["MD5", "SHA-1", "SHA-256"],
        "secure_alternatives": ["HMAC-MD5", "HMAC-SHA1", "HMAC-SHA256"],
        "attack_simulated": "Cryptographic Length Extension",
        "optimizer_used": "Simulated Annealing (SA)"
    })

if __name__ == "__main__":
    print("=" * 60)
    print("  CryptoGuard Upgraded: E-Contract Forgery Detection")
    print("  Server is listening at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)