"""
=============================================================================
Cryptographic Forgery in E-Contracts: A Survey of Metaheuristic
Length-Extension Attacks
-----------------------------------------------------------------------------
Module 5: Hidden Markov Model for E-Contract State Analysis
=============================================================================

Models the e-contract signing workflow as a sequence of discrete states.
A forged contract that has undergone length-extension produces an anomalous
state transition sequence (e.g., unexpected block-boundary state, payload
extension state) detectable by comparing to a trained legitimate HMM.

States:
  0: DRAFT_SUBMISSION      — initial contract upload
  1: HASH_COMPUTATION      — platform computes H(secret || payload)
  2: SIGNATURE_APPLICATION — signer's key applied to hash
  3: PAYLOAD_EXTENSION     — ANOMALOUS: extra data appended post-signature
  4: VERIFICATION_CHECK    — platform verifies submitted (contract, hash)
  5: ARCHIVAL              — signed contract stored

References:
  [34] Wang et al., "HMM for e-contract processing state analysis," Expert Syst. Appl. 2023
"""

import numpy as np
from typing import List, Tuple


# ─── HMM Implementation ───────────────────────────────────────────────────────

class HiddenMarkovModel:
    """
    Discrete Hidden Markov Model with:
      - Baum-Welch training (EM algorithm)
      - Viterbi decoding (most likely state path)
      - Forward algorithm (log-likelihood for anomaly scoring)
    """

    def __init__(self, n_states: int, n_observations: int):
        """
        Args:
            n_states:       Number of hidden states.
            n_observations: Number of distinct observation symbols.
        """
        self.n_states       = n_states
        self.n_observations = n_observations

        # Model parameters (randomly initialized, trained by Baum-Welch)
        rng = np.random.RandomState(42)
        self.pi = self._normalize(rng.dirichlet(np.ones(n_states)))
        self.A  = np.array([self._normalize(rng.dirichlet(np.ones(n_states)))
                            for _ in range(n_states)])
        self.B  = np.array([self._normalize(rng.dirichlet(np.ones(n_observations)))
                            for _ in range(n_states)])

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        s = v.sum()
        return v / s if s > 0 else v

    def _forward(self, O: List[int]) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm: compute alpha matrix and log-likelihood P(O|lambda).

        Returns:
            (alpha, log_likelihood)
        """
        T = len(O)
        alpha = np.zeros((T, self.n_states))

        # Initialization
        alpha[0] = self.pi * self.B[:, O[0]]
        scale    = [alpha[0].sum()]
        alpha[0] /= max(scale[0], 1e-300)

        # Recursion
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * self.B[:, O[t]]
            s = alpha[t].sum()
            scale.append(s)
            alpha[t] /= max(s, 1e-300)

        log_likelihood = sum(np.log(max(s, 1e-300)) for s in scale)
        return alpha, log_likelihood

    def _backward(self, O: List[int], scale: List[float]) -> np.ndarray:
        """Backward algorithm: compute beta matrix."""
        T = len(O)
        beta = np.zeros((T, self.n_states))
        beta[T-1] = 1.0 / max(scale[T-1], 1e-300)

        for t in range(T-2, -1, -1):
            beta[t] = (self.A * self.B[:, O[t+1]] * beta[t+1]).sum(axis=1)
            beta[t] /= max(scale[t], 1e-300)

        return beta

    def log_likelihood(self, O: List[int]) -> float:
        """Compute log P(O | lambda) — anomaly score (more negative = more anomalous)."""
        _, ll = self._forward(O)
        return ll

    def viterbi(self, O: List[int]) -> List[int]:
        """
        Viterbi algorithm: find most likely hidden state sequence.

        Returns:
            List of most probable state indices.
        """
        T = len(O)
        delta = np.zeros((T, self.n_states))
        psi   = np.zeros((T, self.n_states), dtype=int)

        delta[0] = np.log(self.pi + 1e-300) + np.log(self.B[:, O[0]] + 1e-300)

        for t in range(1, T):
            for j in range(self.n_states):
                scores = delta[t-1] + np.log(self.A[:, j] + 1e-300)
                psi[t, j]   = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + np.log(self.B[j, O[t]] + 1e-300)

        # Backtrack
        states    = [0] * T
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def fit(self, sequences: List[List[int]], n_iterations: int = 30):
        """
        Baum-Welch (EM) training on a list of observation sequences.

        Args:
            sequences:    List of observation sequences.
            n_iterations: Number of EM iterations.
        """
        for iteration in range(n_iterations):
            # Accumulators
            A_num  = np.zeros_like(self.A)
            A_den  = np.zeros(self.n_states)
            B_num  = np.zeros_like(self.B)
            B_den  = np.zeros(self.n_states)
            pi_acc = np.zeros(self.n_states)

            for O in sequences:
                T = len(O)
                alpha, ll = self._forward(O)
                scale = [alpha[0].sum()] if T == 1 else []

                # Recompute scale from forward
                a = np.zeros((T, self.n_states))
                a[0] = self.pi * self.B[:, O[0]]
                sc = [a[0].sum()]
                a[0] /= max(sc[0], 1e-300)
                for t in range(1, T):
                    a[t] = (a[t-1] @ self.A) * self.B[:, O[t]]
                    s = a[t].sum()
                    sc.append(s)
                    a[t] /= max(s, 1e-300)

                beta = self._backward(O, sc)

                # Gamma: P(state=i at time t | O, lambda)
                gamma = a * beta
                gamma_sum = gamma.sum(axis=1, keepdims=True)
                gamma /= np.maximum(gamma_sum, 1e-300)

                # Xi: P(state=i at t, state=j at t+1 | O, lambda)
                pi_acc += gamma[0]
                for t in range(T-1):
                    xi_num = (a[t][:, None] * self.A *
                              self.B[:, O[t+1]] * beta[t+1])
                    xi = xi_num / max(xi_num.sum(), 1e-300)
                    A_num += xi
                    A_den += gamma[t]

                A_den += gamma[T-1]
                for t in range(T):
                    B_num[:, O[t]] += gamma[t]
                    B_den += gamma[t]

            # M-step: update parameters
            self.pi = self._normalize(pi_acc)
            for i in range(self.n_states):
                self.A[i]  = self._normalize(A_num[i] / max(A_den[i], 1e-300))
                self.B[i]  = self._normalize(B_num[i] / max(B_den[i], 1e-300))

        return self


# ─── E-Contract State Sequence Generator ──────────────────────────────────────

# Observation symbols: hash/payload events observed during contract processing
OBS = {
    "NORMAL_PAYLOAD":    0,  # Payload within expected length range
    "LONG_PAYLOAD":      1,  # Payload unusually long (possible extension)
    "NORMAL_HASH":       2,  # Hash matches expected format
    "ANOMALOUS_HASH":    3,  # Hash has anomalous entropy / structure
    "PADDING_DETECTED":  4,  # MD padding byte sequence found in payload
    "BLOCK_ALIGNED":     5,  # Payload length is multiple of 64
}

STATES = {
    0: "DRAFT_SUBMISSION",
    1: "HASH_COMPUTATION",
    2: "SIGNATURE_APPLICATION",
    3: "PAYLOAD_EXTENSION",    # ANOMALOUS
    4: "VERIFICATION_CHECK",
    5: "ARCHIVAL",
}


def generate_legitimate_sequences(n: int, seed: int = 42) -> List[List[int]]:
    """Generate legitimate e-contract processing observation sequences."""
    rng = np.random.RandomState(seed)
    seqs = []
    for _ in range(n):
        # Typical flow: normal payload → normal hash → check → archive
        seq = [
            OBS["NORMAL_PAYLOAD"],
            OBS["NORMAL_HASH"],
            rng.choice([OBS["NORMAL_HASH"], OBS["NORMAL_PAYLOAD"]]),
            OBS["NORMAL_HASH"],
            OBS["NORMAL_HASH"],
        ]
        seqs.append(seq)
    return seqs


def generate_forged_sequences(n: int, seed: int = 99) -> List[List[int]]:
    """Generate length-extended forged contract processing sequences."""
    rng = np.random.RandomState(seed)
    seqs = []
    for _ in range(n):
        # Forged flow: normal payload → padding detected → block aligned → anomalous hash
        seq = [
            OBS["NORMAL_PAYLOAD"],
            rng.choice([OBS["PADDING_DETECTED"], OBS["LONG_PAYLOAD"]]),
            OBS["BLOCK_ALIGNED"],
            OBS["LONG_PAYLOAD"],
            OBS["ANOMALOUS_HASH"],
        ]
        seqs.append(seq)
    return seqs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HMM-Based E-Contract State Sequence Analyzer")
    print("  Paper: Cryptographic Forgery in E-Contracts")
    print("=" * 70)

    # Generate datasets
    legit_train  = generate_legitimate_sequences(300, seed=42)
    legit_test   = generate_legitimate_sequences(100, seed=43)
    forged_test  = generate_forged_sequences(100,    seed=99)

    print(f"\n[1] Training HMM on {len(legit_train)} legitimate sequences...")
    hmm = HiddenMarkovModel(n_states=6, n_observations=6)
    hmm.fit(legit_train, n_iterations=20)
    print("    Training complete.")

    # Compute log-likelihoods
    print("\n[2] Log-likelihood distribution:")
    legit_ll  = [hmm.log_likelihood(s) for s in legit_test]
    forged_ll = [hmm.log_likelihood(s) for s in forged_test]
    print(f"   Legitimate  — Mean LL: {np.mean(legit_ll):8.3f}  Std: {np.std(legit_ll):.3f}")
    print(f"   Forged      — Mean LL: {np.mean(forged_ll):8.3f}  Std: {np.std(forged_ll):.3f}")

    # Set threshold: mean - 2*std of legitimate
    threshold = np.mean(legit_ll) - 2.0 * np.std(legit_ll)
    print(f"\n[3] Detection threshold (mean_legit - 2σ): {threshold:.3f}")

    # Evaluate
    all_ll = legit_ll + forged_ll
    all_y  = [0]*len(legit_ll) + [1]*len(forged_ll)
    preds  = [1 if ll < threshold else 0 for ll in all_ll]

    tp = sum(1 for p, y in zip(preds, all_y) if p==1 and y==1)
    fp = sum(1 for p, y in zip(preds, all_y) if p==1 and y==0)
    fn = sum(1 for p, y in zip(preds, all_y) if p==0 and y==1)
    tn = sum(1 for p, y in zip(preds, all_y) if p==0 and y==0)
    dr  = tp / max(tp+fn, 1)
    fpr = fp / max(fp+tn, 1)

    print(f"\n[4] Evaluation Results:")
    print(f"   TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"   Detection Rate (DR)  : {dr*100:.1f}%")
    print(f"   False Positive Rate  : {fpr*100:.1f}%")
    print(f"   Accuracy             : {(tp+tn)/(tp+fp+fn+tn)*100:.1f}%")

    # Viterbi decoding example
    print("\n[5] Viterbi State Path Decoding:")
    print("-" * 50)
    ex_legit  = legit_test[0]
    ex_forged = forged_test[0]
    path_legit  = hmm.viterbi(ex_legit)
    path_forged = hmm.viterbi(ex_forged)

    obs_names = {v: k for k, v in OBS.items()}
    print(f"   Legitimate contract:")
    print(f"     Observations : {[obs_names[o] for o in ex_legit]}")
    print(f"     Viterbi path : {[STATES[s] for s in path_legit]}")
    print(f"     Log-likelihood: {hmm.log_likelihood(ex_legit):.3f}")

    print(f"\n   Forged contract (length-extended):")
    print(f"     Observations : {[obs_names[o] for o in ex_forged]}")
    print(f"     Viterbi path : {[STATES[s] for s in path_forged]}")
    print(f"     Log-likelihood: {hmm.log_likelihood(ex_forged):.3f}")

    if OBS["PADDING_DETECTED"] in ex_forged or OBS["BLOCK_ALIGNED"] in ex_forged:
        print(f"\n   [ALERT] Padding/block-alignment observation detected — possible forgery.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
