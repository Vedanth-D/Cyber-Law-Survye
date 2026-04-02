from flask import Flask, request, jsonify, render_template, make_response
from detector import (
    vulnerable_sign, secure_sign,
    length_extension_attack, detect_forgery,
    simulated_annealing_secret_length, md_padding
)

app = Flask(__name__)
SECRET_KEY = "mysecretkey2024"


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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/sign/vulnerable", methods=["GET", "POST", "OPTIONS"])
def sign_vulnerable():
    if request.method in ("GET", "OPTIONS"):
        return jsonify({"status": "endpoint ready"})
    data    = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    tag = vulnerable_sign(SECRET_KEY, message)
    return jsonify({
        "method":  "Raw SHA-256 (Vulnerable)",
        "message": message,
        "tag":     tag,
        "warning": "This MAC is vulnerable to hash length-extension attacks!"
    })


@app.route("/api/sign/secure", methods=["GET", "POST", "OPTIONS"])
def sign_secure():
    if request.method in ("GET", "OPTIONS"):
        return jsonify({"status": "endpoint ready"})
    data    = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    tag = secure_sign(SECRET_KEY, message)
    return jsonify({
        "method":  "HMAC-SHA256 (Secure)",
        "message": message,
        "tag":     tag,
        "info":    "HMAC double-hashing prevents length-extension attacks."
    })


@app.route("/api/attack/length-extension", methods=["GET", "POST", "OPTIONS"])
def attack():
    if request.method in ("GET", "OPTIONS"):
        return jsonify({"status": "endpoint ready"})
    data         = request.get_json(force=True, silent=True) or {}
    original_tag = data.get("tag", "").strip()
    original_msg = data.get("message", "").strip()
    extension    = data.get("extension", " | FORGED CLAUSE: Pay attacker $9999")
    if not original_tag or not original_msg:
        return jsonify({"error": "tag and message are required"}), 400
    guessed_len, sa_log = simulated_annealing_secret_length(original_tag)
    padding_hex         = md_padding(guessed_len + len(original_msg)).hex()
    forged_tag, forged_msg = length_extension_attack(
        original_tag, original_msg, extension, guessed_len
    )
    return jsonify({
        "attack":                   "Hash Length-Extension Attack",
        "original_tag":             original_tag,
        "original_message":         original_msg,
        "extension_appended":       extension,
        "sa_guessed_secret_length": guessed_len,
        "md_padding_hex":           padding_hex[:64] + "...",
        "forged_tag":               forged_tag,
        "forged_message_display":   forged_msg,
        "sa_convergence_log":       sa_log,
        "result":                   "Forged tag crafted WITHOUT knowing the secret key!"
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
    method  = data.get("method", "secure")
    if not message or not tag:
        return jsonify({"error": "message and tag are required"}), 400
    expected = vulnerable_sign(SECRET_KEY, message) if method == "vulnerable" \
               else secure_sign(SECRET_KEY, message)
    valid = (expected == tag)
    return jsonify({
        "valid":         valid,
        "method":        method,
        "submitted_tag": tag,
        "expected_tag":  expected,
        "verdict":       "VALID CONTRACT" if valid else "INVALID / TAMPERED CONTRACT"
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "model": "Random Forest (100 estimators)",
        "training_samples": 1200,
        "features": 8,
        "hash_functions": ["MD5", "SHA-1", "SHA-256"],
        "secure_alternatives": ["HMAC-SHA256", "SHA-3 / Keccak", "BLAKE3"],
        "attack_simulated": "Merkle-Damgard Length Extension",
        "optimizer_used": "Simulated Annealing (SA)"
    })


if __name__ == "__main__":
    print("=" * 55)
    print("  E-Contract Cryptographic Forgery Detection System")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="127.0.0.1", port=5000)