"""
app.py  —  Flask backend for E-Contract Forgery Detection System
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from core.crypto_engine import (
    vulnerable_sign, vulnerable_verify,
    secure_sign, secure_verify,
    perform_attack, md_padding
)
from core.detection_engine import (
    extract_features, EntropyDetector, StaticThresholdDetector,
    RandomForest, LSTMAutoencoder, HMMDetector,
    PSOOptimizer, GWOOptimizer,
    generate_dataset, normalize, evaluate_all,
    OBS_NAMES, STATE_NAMES, contract_to_obs_fn
)
import hashlib, json, numpy as np

app = Flask(__name__)

# ── Pre-train models once at startup ─────────────────────────────────────────
print("[boot] Generating dataset and training models...")
X, y = generate_dataset(400, 400, seed=42)
split = int(.8*len(y))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]
X_tr_n, _mn, _mx = normalize(X_tr.copy())
def _norm(v): return (v - _mn) / np.maximum(_mx-_mn, 1.0)
X_te_n = _norm(X_te)

_ed = EntropyDetector().fit(X_tr, y_tr)
_rf = RandomForest(50, seed=42).fit(X_tr_n, y_tr)
_lstm = LSTMAutoencoder(8,16,42)
_legit_seqs = [X_tr_n[i:i+10] for i in range(0, min(200,len(X_tr_n))-10, 10) if y_tr[i]==0]
_lstm.fit_threshold(_legit_seqs if _legit_seqs else [X_tr_n[:10]], k=2.0)
_hmm = HMMDetector(6,6).fit(
    [[0,2,2,2,0] for _ in range(100)], n_iter=10
)
_hmm.fit_threshold([[0,2,2,2,0] for _ in range(50)], k=2.0)
print("[boot] Models ready.")

SECRET_KEY = b"JAIN_ECONTRACT_SECRET_2024"


def _detect_contract(contract_bytes: bytes, sig: str) -> dict:
    """Run all detectors on a submitted contract and return per-detector verdicts."""
    fv = extract_features(contract_bytes, sig)
    fv_n = _norm(fv.reshape(1,-1))[0]

    # Static threshold
    static_flag = bool((fv[0]>800) or (fv[3]>0.12))

    # Entropy
    entropy_score = _ed.score_single(fv)
    entropy_flag  = bool(entropy_score > 0)

    # RF
    rf_prob = float(_rf.score_single(fv_n))
    rf_flag = rf_prob >= 0.5

    # LSTM
    seq = np.tile(fv_n, (10,1))
    lstm_label, lstm_err = _lstm.predict_seq(seq)
    lstm_flag = bool(lstm_label)

    # HMM
    n = int(fv[0])
    obs = [
        1 if n > 600 else 0,
        4 if fv[5] > 0.5 else 2,
        4 if fv[3] > 0.10 else 2,
        3 if fv[2] < 3.0 else 2,
        1 if n > 500 else 0,
    ]
    hmm_label, hmm_ll, hmm_path = _hmm.predict_seq(obs)
    hmm_flag = bool(hmm_label)

    votes = sum([static_flag, entropy_flag, rf_flag, lstm_flag, hmm_flag])
    overall = "FORGED" if votes >= 3 else "LEGITIMATE"

    return {
        "overall": overall,
        "confidence": round(votes/5*100),
        "votes": votes,
        "features": {n: round(float(v),4) for n,v in zip(
            ["payload_length","entropy","padding_entropy","null_byte_ratio",
             "high_byte_ratio","block_boundary","length_mod_64","hash_entropy"], fv)},
        "detectors": {
            "static_threshold": {"flag": static_flag, "reason": f"Length={int(fv[0])}, NullRatio={fv[3]:.3f}"},
            "entropy_anomaly":  {"flag": entropy_flag, "score": round(entropy_score,4),
                                  "reason": f"PadEntropy={fv[2]:.2f}, NullRatio={fv[3]:.3f}"},
            "random_forest":    {"flag": rf_flag,  "probability": round(rf_prob,3)},
            "lstm_autoencoder": {"flag": lstm_flag, "reconstruction_error": round(lstm_err,5),
                                  "threshold": round(float(_lstm.threshold or 0),5)},
            "hmm_state":        {"flag": hmm_flag, "log_likelihood": round(hmm_ll,3),
                                  "state_path": [STATE_NAMES[s] for s in hmm_path],
                                  "observations": [OBS_NAMES[o] for o in obs]},
        }
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sign", methods=["POST"])
def api_sign():
    data = request.json
    contract = data.get("contract","").encode()
    mode = data.get("mode","secure")
    if mode == "vulnerable":
        sig = vulnerable_sign(SECRET_KEY, contract)
    else:
        sig = secure_sign(SECRET_KEY, contract)
    return jsonify({"signature": sig, "mode": mode,
                    "contract_length": len(contract),
                    "algorithm": "SHA-256(secret||contract)" if mode=="vulnerable" else "HMAC-SHA256"})

@app.route("/api/verify", methods=["POST"])
def api_verify():
    data = request.json
    contract = data.get("contract","").encode()
    sig = data.get("signature","")
    mode = data.get("mode","secure")
    if mode == "vulnerable":
        valid = vulnerable_verify(SECRET_KEY, contract, sig)
    else:
        valid = secure_verify(SECRET_KEY, contract, sig)
    return jsonify({"valid": valid, "mode": mode})

@app.route("/api/attack", methods=["POST"])
def api_attack():
    data = request.json
    contract  = data.get("contract","").encode()
    signature = data.get("signature","")
    malicious = data.get("malicious","").encode()
    if not signature:
        signature = vulnerable_sign(SECRET_KEY, contract)
    result = perform_attack(contract, signature, malicious, len(SECRET_KEY))
    # Verify forged passes vulnerable check
    result["forged_passes_vulnerable"] = vulnerable_verify(SECRET_KEY, result["forged_contract_hex"].encode(), result["forged_signature"])
    result["forged_passes_hmac"]       = secure_verify(SECRET_KEY, bytes.fromhex(result["forged_contract_hex"]) if all(c in '0123456789abcdef' for c in result["forged_contract_hex"]) else b"", result["forged_signature"])
    # Re-check properly
    forged_bytes = contract + bytes.fromhex(result["padding_hex"]) + malicious
    result["forged_passes_vulnerable"] = vulnerable_verify(SECRET_KEY, forged_bytes, result["forged_signature"])
    result["forged_passes_hmac"]       = secure_verify(SECRET_KEY, forged_bytes, result["forged_signature"])
    return jsonify(result)

@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.json
    contract  = data.get("contract","").encode()
    signature = data.get("signature","")
    if not signature:
        signature = vulnerable_sign(SECRET_KEY, contract)
    result = _detect_contract(contract, signature)
    return jsonify(result)

@app.route("/api/evaluate", methods=["GET"])
def api_evaluate():
    res = evaluate_all()
    return jsonify(res)

@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    data = request.json
    algo = data.get("algorithm","pso")
    n_iter = int(data.get("n_iter", 80))
    # Generate scores for demo
    rng = np.random.RandomState(7)
    scores = np.concatenate([rng.normal(.3,.1,300).clip(0,1), rng.normal(.75,.12,300).clip(0,1)])
    labels = np.array([0]*300+[1]*300)

    if algo == "pso":
        opt = PSOOptimizer(30, n_iter, seed=42)
    else:
        opt = GWOOptimizer(30, n_iter, seed=42)

    best_t, best_f, history = opt.optimize(scores, labels)

    preds = (scores >= best_t).astype(int)
    tp=int(((preds==1)&(labels==1)).sum()); fp=int(((preds==1)&(labels==0)).sum())
    fn=int(((preds==0)&(labels==1)).sum()); tn=int(((preds==0)&(labels==0)).sum())
    dr=round(tp/max(tp+fn,1)*100,1); fpr=round(fp/max(fp+tn,1)*100,1)

    return jsonify({
        "algorithm": algo.upper(),
        "best_threshold": round(float(best_t),4),
        "best_fitness": round(float(best_f),4),
        "DR": dr, "FPR": fpr,
        "history": [round(v,4) for v in history],
        "convergence_points": list(range(len(history)))
    })

if __name__ == "__main__":
    app.run(debug=False, port=5050)
