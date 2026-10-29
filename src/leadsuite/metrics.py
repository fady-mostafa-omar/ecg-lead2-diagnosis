"""Scores for the multi-label PTB-XL tasks: per-class ROC AUC, macro AUC, and a patient-level bootstrap."""
import numpy as np
from sklearn.metrics import roc_auc_score

from .data import SUPER


def auc_per_class(Y, P):
    out = {c: roc_auc_score(Y[:, k], P[:, k]) for k, c in enumerate(SUPER)}
    out["macro"] = float(np.mean([out[c] for c in SUPER]))
    return out


def macro_auc(Y, P):
    """Mean ROC AUC over the columns that have both positives and negatives (any number of classes). For the 5
    superclasses it equals auc_per_class(Y, P)["macro"]."""
    cols = [k for k in range(Y.shape[1]) if 0 < Y[:, k].sum() < len(Y)]
    return float(np.mean([roc_auc_score(Y[:, k], P[:, k]) for k in cols]))


def patient_bootstrap(Y, preds, patients, n=1000, seed=0):
    """preds: {name: P}. Resamples patients (all ECGs of a patient together). Returns {name: [n, 6]} AUC draws
    (5 classes + macro), for confidence intervals and paired differences."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(patients, return_inverse=True)
    rows = [np.where(inv == k)[0] for k in range(len(uniq))]
    draws = {k: [] for k in preds}
    for _ in range(n):
        idx = np.concatenate([rows[j] for j in rng.integers(0, len(uniq), len(uniq))])
        if any(Y[idx, c].min() == Y[idx, c].max() for c in range(Y.shape[1])):
            continue
        for k, P in preds.items():
            a = auc_per_class(Y[idx], P[idx])
            draws[k].append([a[c] for c in SUPER] + [a["macro"]])
    return {k: np.array(v) for k, v in draws.items()}
