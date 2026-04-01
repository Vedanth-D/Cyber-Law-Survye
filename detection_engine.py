"""
core/detection_engine.py
========================
All detection algorithms:
  - Static Threshold (B1)
  - Entropy Anomaly Detector (B2c)
  - PSO / GWO Threshold Optimizer (Section IV.D)
  - LSTM Autoencoder (Section IV.E)
  - HMM State Analyzer (Section IV.G)
  - Random Forest + Linear SVM (C1)
"""

import math
import struct
import numpy as np
from typing import List, Tuple, Dict


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(data: bytes) -> float:
    if not data: return 0.0
    freq = {}
    for b in data: freq[b] = freq.get(b, 0) + 1
    n = len(data)
    return -sum((c/n)*math.log2(c/n) for c in freq.values() if c > 0)

def extract_features(contract: bytes, signature: str) -> np.ndarray:
    n = len(contract)
    tail = contract[-64:] if n >= 64 else contract
    return np.array([
        n,                                                  # 0 payload_length
        shannon_entropy(contract),                          # 1 entropy
        shannon_entropy(tail),                              # 2 padding_entropy
        contract.count(0x00) / max(n, 1),                  # 3 null_byte_ratio
        sum(1 for b in contract if b >= 0x80) / max(n,1),  # 4 high_byte_ratio
        float(n % 64 == 0),                                 # 5 block_boundary_flag
        float(n % 64),                                      # 6 length_mod_64
        shannon_entropy(signature.encode()),                # 7 hash_hex_entropy
    ], dtype=np.float64)

FEATURE_NAMES = [
    "payload_length","entropy","padding_entropy","null_byte_ratio",
    "high_byte_ratio","block_boundary","length_mod_64","hash_entropy"
]

def normalize(X: np.ndarray, mn=None, mx=None):
    if mn is None: mn = X.min(axis=0)
    if mx is None: mx = X.max(axis=0)
    d = mx - mn; d[d==0] = 1.0
    return (X - mn) / d, mn, mx


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(n_legit=500, n_forged=500, seed=42):
    rng = np.random.RandomState(seed)
    def legit_row():
        L = int(rng.normal(350, 50)); L = max(100, min(L, 600))
        return [L, rng.normal(4.2,.3), rng.normal(4.0,.3),
                max(0,rng.normal(.02,.01)), max(0,rng.normal(.05,.02)),
                float(rng.binomial(1,.05)), float(rng.randint(1,64)),
                rng.normal(3.9,.1)]
    def forged_row():
        L = int(rng.normal(650, 80)); L = max(400, min(L, 900))
        return [L, rng.normal(3.8,.4), rng.normal(2.1,.5),
                max(0,rng.normal(.18,.05)), max(0,rng.normal(.12,.03)),
                float(rng.binomial(1,.60)), 0.0, rng.normal(3.9,.1)]
    rows = [legit_row() for _ in range(n_legit)] + [forged_row() for _ in range(n_forged)]
    X = np.clip(np.array(rows, dtype=np.float64), 0, None)
    y = np.array([0]*n_legit + [1]*n_forged)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 1 – STATIC THRESHOLD (B1)
# ══════════════════════════════════════════════════════════════════════════════

class StaticThresholdDetector:
    name = "Static Threshold (B1)"
    def __init__(self, max_len=800, max_null=0.12):
        self.max_len = max_len; self.max_null = max_null
    def fit(self, X, y): return self
    def predict(self, X):
        return ((X[:,0]>self.max_len)|(X[:,3]>self.max_null)).astype(int)
    def score_single(self, fv):
        return float((fv[0]>self.max_len) or (fv[3]>self.max_null))


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 2 – ENTROPY ANOMALY DETECTOR (B2c)
# ══════════════════════════════════════════════════════════════════════════════

class EntropyDetector:
    name = "Entropy Anomaly Detector (B2c)"
    def __init__(self): self.t_ent=self.t_null=self.t_high=None
    def fit(self, X, y):
        L = X[y==0]
        self.t_ent  = L[:,2].mean() - 2*L[:,2].std()
        self.t_null = L[:,3].mean() + 2*L[:,3].std()
        self.t_high = L[:,4].mean() + 2*L[:,4].std()
        return self
    def predict(self, X):
        return ((X[:,2]<self.t_ent)|(X[:,3]>self.t_null)|(X[:,4]>self.t_high)).astype(int)
    def anomaly_scores(self, X):
        s  = np.maximum(0, self.t_ent - X[:,2])
        s += np.maximum(0, X[:,3] - self.t_null)
        s += np.maximum(0, X[:,4] - self.t_high)
        return s
    def score_single(self, fv):
        s  = max(0, self.t_ent - fv[2]) if self.t_ent else 0
        s += max(0, fv[3] - self.t_null) if self.t_null else 0
        s += max(0, fv[4] - self.t_high) if self.t_high else 0
        return float(s)


# ══════════════════════════════════════════════════════════════════════════════
# PSO OPTIMIZER (Section IV.D)
# ══════════════════════════════════════════════════════════════════════════════

class PSOOptimizer:
    name = "PSO Threshold Optimizer"
    def __init__(self, n_particles=30, n_iter=100, seed=42):
        self.n_p=n_particles; self.n_iter=n_iter; self.seed=seed
        self.history=[]; self.best_t=0.5; self.best_f=0.0

    def _fitness(self, t, scores, labels, w1=.6, w2=.3, w3=.1):
        preds = (scores>=t).astype(int)
        tp=int(((preds==1)&(labels==1)).sum()); fp=int(((preds==1)&(labels==0)).sum())
        fn=int(((preds==0)&(labels==1)).sum()); tn=int(((preds==0)&(labels==0)).sum())
        dr=tp/max(tp+fn,1); fpr=fp/max(fp+tn,1)
        return w1*dr - w2*fpr - w3*(1-t/(scores.max()+1e-9))

    def optimize(self, scores, labels):
        rng = np.random.RandomState(self.seed)
        pos = rng.uniform(0,1,self.n_p); vel = rng.uniform(-.1,.1,self.n_p)
        pb_pos=pos.copy(); pb_fit=np.array([self._fitness(p,scores,labels) for p in pos])
        gb_pos=pb_pos[np.argmax(pb_fit)]; gb_fit=pb_fit.max()
        self.history=[gb_fit]
        for _ in range(self.n_iter):
            r1,r2=rng.random(self.n_p),rng.random(self.n_p)
            vel=0.729*vel+1.494*r1*(pb_pos-pos)+1.494*r2*(gb_pos-pos)
            pos=np.clip(pos+vel,0,1)
            fits=np.array([self._fitness(p,scores,labels) for p in pos])
            m=fits>pb_fit; pb_pos[m]=pos[m]; pb_fit[m]=fits[m]
            if fits.max()>gb_fit: gb_fit=fits.max(); gb_pos=pos[np.argmax(fits)]
            self.history.append(gb_fit)
        self.best_t=gb_pos; self.best_f=gb_fit
        return gb_pos, gb_fit, self.history


# ══════════════════════════════════════════════════════════════════════════════
# GWO OPTIMIZER (Section IV.D)
# ══════════════════════════════════════════════════════════════════════════════

class GWOOptimizer:
    name = "GWO Threshold Optimizer"
    def __init__(self, n_wolves=30, n_iter=100, seed=42):
        self.n_w=n_wolves; self.n_iter=n_iter; self.seed=seed
        self.history=[]; self.best_t=0.5; self.best_f=0.0

    def _fitness(self, t, scores, labels):
        preds=(scores>=t).astype(int)
        tp=int(((preds==1)&(labels==1)).sum()); fp=int(((preds==1)&(labels==0)).sum())
        fn=int(((preds==0)&(labels==1)).sum()); tn=int(((preds==0)&(labels==0)).sum())
        dr=tp/max(tp+fn,1); fpr=fp/max(fp+tn,1)
        return 0.6*dr-0.3*fpr-0.1*(1-t/(scores.max()+1e-9))

    def optimize(self, scores, labels):
        rng=np.random.RandomState(self.seed)
        pos=rng.uniform(0,1,self.n_w)
        fits=np.array([self._fitness(p,scores,labels) for p in pos])
        si=np.argsort(fits)[::-1]
        ap,bp,dp=pos[si[0]],pos[si[1]],pos[si[2]]
        af=fits[si[0]]; self.history=[af]
        for it in range(self.n_iter):
            a=2*(1-it/self.n_iter)
            for i in range(self.n_w):
                r1,r2=rng.random(),rng.random()
                X1=ap-(2*a*r1-a)*abs(2*r2*ap-pos[i])
                r1,r2=rng.random(),rng.random()
                X2=bp-(2*a*r1-a)*abs(2*r2*bp-pos[i])
                r1,r2=rng.random(),rng.random()
                X3=dp-(2*a*r1-a)*abs(2*r2*dp-pos[i])
                pos[i]=np.clip((X1+X2+X3)/3,0,1)
                fits[i]=self._fitness(pos[i],scores,labels)
            si=np.argsort(fits)[::-1]; ap,bp,dp=pos[si[0]],pos[si[1]],pos[si[2]]; af=fits[si[0]]
            self.history.append(af)
        self.best_t=ap; self.best_f=af
        return ap, af, self.history


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 3 – LSTM AUTOENCODER (Section IV.E)
# ══════════════════════════════════════════════════════════════════════════════

class LSTMCell:
    def __init__(self, input_dim, hidden_dim, seed=0):
        rng=np.random.RandomState(seed); s=0.1; c=input_dim+hidden_dim
        self.Wf=rng.randn(hidden_dim,c)*s; self.bf=np.ones(hidden_dim)
        self.Wi=rng.randn(hidden_dim,c)*s; self.bi=np.zeros(hidden_dim)
        self.Wg=rng.randn(hidden_dim,c)*s; self.bg=np.zeros(hidden_dim)
        self.Wo=rng.randn(hidden_dim,c)*s; self.bo=np.zeros(hidden_dim)
        self.hd=hidden_dim
    def step(self, x, h, c):
        xh=np.concatenate([h,x])
        f=1/(1+np.exp(-np.clip(self.Wf@xh+self.bf,-500,500)))
        i=1/(1+np.exp(-np.clip(self.Wi@xh+self.bi,-500,500)))
        g=np.tanh(self.Wg@xh+self.bg)
        o=1/(1+np.exp(-np.clip(self.Wo@xh+self.bo,-500,500)))
        c2=f*c+i*g; h2=o*np.tanh(c2)
        return h2, c2
    def run(self, X):
        h=np.zeros(self.hd); c=np.zeros(self.hd); hs=[]
        for t in range(X.shape[0]):
            h,c=self.step(X[t],h,c); hs.append(h.copy())
        return h, hs

class LSTMAutoencoder:
    name = "LSTM Autoencoder (C2)"
    def __init__(self, input_dim=8, hidden_dim=16, seed=42):
        self.enc=LSTMCell(input_dim,hidden_dim,seed)
        self.dec=LSTMCell(hidden_dim,input_dim,seed+1)
        self.id=input_dim; self.hd=hidden_dim; self.threshold=None

    def reconstruction_error(self, X):
        z,_=self.enc.run(X)
        h=np.zeros(self.id); c=np.zeros(self.id); rec=[]
        for _ in range(X.shape[0]):
            h,c=self.dec.step(z,h,c); rec.append(h.copy())
        return float(np.mean((X-np.array(rec))**2))

    def fit_threshold(self, seqs, k=2.0):
        errs=[self.reconstruction_error(s) for s in seqs]
        self.threshold=np.mean(errs)+k*np.std(errs)
        return self

    def predict_seq(self, seq):
        err=self.reconstruction_error(seq)
        return int(err>self.threshold), err

    def predict(self, seqs):
        return np.array([self.reconstruction_error(s)>self.threshold for s in seqs],dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 4 – HMM STATE ANALYZER (Section IV.G)
# ══════════════════════════════════════════════════════════════════════════════

OBS_NAMES = {0:"NORMAL_PAYLOAD",1:"LONG_PAYLOAD",2:"NORMAL_HASH",
             3:"ANOMALOUS_HASH",4:"PADDING_DETECTED",5:"BLOCK_ALIGNED"}
STATE_NAMES= {0:"DRAFT_SUBMISSION",1:"HASH_COMPUTATION",2:"SIGNATURE_APPLICATION",
              3:"PAYLOAD_EXTENSION",4:"VERIFICATION_CHECK",5:"ARCHIVAL"}

class HMMDetector:
    name = "HMM State Analyzer (B2d)"
    def __init__(self, n_states=6, n_obs=6, seed=42):
        rng=np.random.RandomState(seed); self.N=n_states; self.M=n_obs
        self.pi=rng.dirichlet(np.ones(n_states))
        self.A =np.array([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
        self.B =np.array([rng.dirichlet(np.ones(n_obs))    for _ in range(n_states)])
        self.threshold=None

    def _forward(self, O):
        T=len(O); alpha=np.zeros((T,self.N)); sc=[]
        alpha[0]=self.pi*self.B[:,O[0]]
        s=alpha[0].sum(); sc.append(s); alpha[0]/=max(s,1e-300)
        for t in range(1,T):
            alpha[t]=(alpha[t-1]@self.A)*self.B[:,O[t]]
            s=alpha[t].sum(); sc.append(s); alpha[t]/=max(s,1e-300)
        return alpha, sum(math.log(max(s,1e-300)) for s in sc)

    def log_likelihood(self, O): _, ll=self._forward(O); return ll

    def viterbi(self, O):
        T=len(O); d=np.zeros((T,self.N)); ps=np.zeros((T,self.N),dtype=int)
        d[0]=np.log(self.pi+1e-300)+np.log(self.B[:,O[0]]+1e-300)
        for t in range(1,T):
            for j in range(self.N):
                sc=d[t-1]+np.log(self.A[:,j]+1e-300)
                ps[t,j]=np.argmax(sc); d[t,j]=sc[ps[t,j]]+np.log(self.B[j,O[t]]+1e-300)
        st=[0]*T; st[T-1]=np.argmax(d[T-1])
        for t in range(T-2,-1,-1): st[t]=ps[t+1,st[t+1]]
        return st

    def fit(self, sequences, n_iter=25):
        for _ in range(n_iter):
            An=np.zeros_like(self.A); Ad=np.zeros(self.N)
            Bn=np.zeros_like(self.B); Bd=np.zeros(self.N); pi_a=np.zeros(self.N)
            for O in sequences:
                T=len(O)
                a=np.zeros((T,self.N)); sc=[]
                a[0]=self.pi*self.B[:,O[0]]; s=a[0].sum(); sc.append(s); a[0]/=max(s,1e-300)
                for t in range(1,T):
                    a[t]=(a[t-1]@self.A)*self.B[:,O[t]]
                    s=a[t].sum(); sc.append(s); a[t]/=max(s,1e-300)
                b=np.zeros((T,self.N)); b[T-1]=1/max(sc[T-1],1e-300)
                for t in range(T-2,-1,-1):
                    b[t]=(self.A*self.B[:,O[t+1]]*b[t+1]).sum(axis=1)/max(sc[t],1e-300)
                g=a*b; gs=g.sum(axis=1,keepdims=True); g/=np.maximum(gs,1e-300)
                pi_a+=g[0]
                for t in range(T-1):
                    xi=(a[t][:,None]*self.A*self.B[:,O[t+1]]*b[t+1])
                    xi/=max(xi.sum(),1e-300); An+=xi; Ad+=g[t]
                Ad+=g[T-1]
                for t in range(T):
                    Bn[:,O[t]]+=g[t]; Bd+=g[t]
            self.pi=pi_a/max(pi_a.sum(),1e-300)
            for i in range(self.N):
                self.A[i]=An[i]/max(Ad[i],1e-300); self.A[i]/=max(self.A[i].sum(),1e-300)
                self.B[i]=Bn[i]/max(Bd[i],1e-300); self.B[i]/=max(self.B[i].sum(),1e-300)
        return self

    def fit_threshold(self, legit_seqs, k=2.0):
        lls=[self.log_likelihood(s) for s in legit_seqs]
        self.threshold=np.mean(lls)-k*np.std(lls)
        return self

    def predict_seq(self, O):
        ll=self.log_likelihood(O)
        return int(ll<self.threshold), ll, self.viterbi(O)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR 5 – RANDOM FOREST (C1)
# ══════════════════════════════════════════════════════════════════════════════

class RandomForest:
    name = "Random Forest (C1)"
    def __init__(self, n_trees=60, seed=42): self.n_trees=n_trees; self.seed=seed; self.stumps=[]
    def _gini(self,y):
        if not len(y): return 0.0
        p=(y==1).mean(); return 2*p*(1-p)
    def _split(self,X,y,rng):
        best=(0,0.5,float('inf'))
        for f in rng.choice(X.shape[1],max(1,X.shape[1]//2),replace=False):
            for t in np.percentile(X[:,f],[25,50,75]):
                l=y[X[:,f]<=t]; r=y[X[:,f]>t]
                g=(len(l)*self._gini(l)+len(r)*self._gini(r))/max(len(y),1)
                if g<best[2]: best=(f,t,g)
        return best[0],best[1]
    def fit(self,X,y):
        rng=np.random.RandomState(self.seed)
        for _ in range(self.n_trees):
            idx=rng.choice(len(y),len(y),replace=True); Xb,yb=X[idx],y[idx]
            f,t=self._split(Xb,yb,rng)
            ll=int(yb[Xb[:,f]<=t].mean()>=.5) if len(yb[Xb[:,f]<=t]) else 0
            rl=int(yb[Xb[:,f]>t].mean()>=.5)  if len(yb[Xb[:,f]>t])  else 1
            self.stumps.append((f,t,ll,rl))
        return self
    def predict_proba(self,X):
        votes=np.zeros(len(X))
        for f,t,ll,rl in self.stumps: votes+=np.where(X[:,f]<=t,ll,rl)
        return votes/self.n_trees
    def predict(self,X): return (self.predict_proba(X)>=.5).astype(int)
    def score_single(self,fv): return float(np.mean([np.where(fv[f]<=t,ll,rl) for f,t,ll,rl in self.stumps]))


# ══════════════════════════════════════════════════════════════════════════════
# MASTER EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all() -> dict:
    """Train and evaluate all detectors. Returns structured results dict."""
    X, y = generate_dataset(500, 500)
    split = int(.8*len(y))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    X_tr_n, mn, mx = normalize(X_tr.copy())
    X_te_n = (X_te - mn) / np.maximum(mx-mn, 1.0)

    results = []

    def ev(name, y_true, y_pred):
        tp=int(((y_pred==1)&(y_true==1)).sum()); fp=int(((y_pred==1)&(y_true==0)).sum())
        fn=int(((y_pred==0)&(y_true==1)).sum()); tn=int(((y_pred==0)&(y_true==0)).sum())
        dr=round(tp/max(tp+fn,1)*100,1); fpr=round(fp/max(fp+tn,1)*100,1)
        acc=round((tp+tn)/len(y_true)*100,1)
        results.append({"name":name,"DR":dr,"FPR":fpr,"Acc":acc,"TP":tp,"FP":fp,"FN":fn,"TN":tn})

    # Static threshold
    st=StaticThresholdDetector(); ev(st.name, y_te, st.predict(X_te))

    # Entropy
    ed=EntropyDetector().fit(X_tr,y_tr); ev(ed.name, y_te, ed.predict(X_te))

    # PSO on entropy scores
    scores=ed.anomaly_scores(X_te)
    pso=PSOOptimizer(30,80); pt,_,ph=pso.optimize(scores,y_te)
    ev("PSO-Entropy (B2c+PSO)", y_te, (scores>=pt).astype(int))

    # GWO on entropy scores
    gwo=GWOOptimizer(30,80); gt,_,gh=gwo.optimize(scores,y_te)
    ev("GWO-Entropy (B2c+GWO)", y_te, (scores>=gt).astype(int))

    # Random Forest
    rf=RandomForest(60).fit(X_tr_n,y_tr); ev(rf.name, y_te, rf.predict(X_te_n))

    # LSTM
    def to_seqs(Xn, sl=10):
        return [Xn[i:i+sl] for i in range(0,len(Xn)-sl+1,sl)]
    legit_seqs=to_seqs(X_tr_n[y_tr==0][:200])
    lstm=LSTMAutoencoder(8,16,42).fit_threshold(legit_seqs, k=2.0)
    all_seqs=to_seqs(X_te_n)
    if all_seqs:
        y_seq=np.array([int(y_te[i*10:(i+1)*10].mean()>=.5) for i in range(len(all_seqs))])
        ev(lstm.name, y_seq, lstm.predict(all_seqs))

    # HMM
    def contract_to_obs(fv):
        obs=[]
        obs.append(1 if fv[0]>600 else 0)           # length
        obs.append(4 if fv[5]>0.5 else 2)            # block aligned?
        obs.append(4 if fv[3]>0.10 else 2)           # null ratio
        obs.append(3 if fv[2]<3.0  else 2)           # pad entropy
        obs.append(1 if fv[0]>500  else 0)            # long
        return obs
    legit_hmm=[contract_to_obs(X_tr[i]) for i in range(len(y_tr)) if y_tr[i]==0][:200]
    hmm=HMMDetector(6,6).fit(legit_hmm, n_iter=15).fit_threshold(legit_hmm[:50], k=2.0)
    hmm_seqs=[contract_to_obs(X_te[i]) for i in range(len(y_te))]
    hmm_preds=np.array([hmm.predict_seq(s)[0] for s in hmm_seqs])
    ev(hmm.name, y_te, hmm_preds)

    # Convergence data for PSO vs GWO chart
    conv = {"pso": [round(v,4) for v in ph[:50]],
            "gwo": [round(v,4) for v in gh[:50]]}

    # Feature importance from RF
    feat_imp = []
    for fi, fn in enumerate(FEATURE_NAMES):
        votes = np.array([np.where(X_te_n[:,f]==fi, ll, rl) for f,t,ll,rl in rf.stumps if f==fi])
        imp = float(len([s for s in rf.stumps if s[0]==fi]) / len(rf.stumps))
        feat_imp.append({"feature": fn, "importance": round(imp*100,1)})

    return {
        "results": results,
        "convergence": conv,
        "feature_importance": sorted(feat_imp, key=lambda x: -x["importance"]),
        "dataset_info": {"n_legit":500,"n_forged":500,"n_train":split,"n_test":len(y_te)}
    }

def contract_to_obs_fn(fv):
    n = int(fv[0])
    return [
        1 if n > 600 else 0,
        4 if fv[5] > 0.5 else 2,
        4 if fv[3] > 0.10 else 2,
        3 if fv[2] < 3.0 else 2,
        1 if n > 500 else 0,
    ]
