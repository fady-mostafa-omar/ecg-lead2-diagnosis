# STEP 3.5 — Signal-quality gate for lead II: judge each 10-s strip from the strip alone and refuse to diagnose
# strips that are too noisy ("signal too noisy, check the electrodes") instead of giving a wrong answer.
#   Gate 1 (signal-based, used for all three projects): 7 fast features + a regressor that estimates the SNR in dB.
#   Gate 2 (model-based, for comparison): disagreement between the 5 networks of the noise-trained ensemble.
# Leakage-free: the estimator is trained on training-fold ECGs + training-part noise; its threshold is chosen on the
# validation fold (also training-part noise); everything is then scored once on the test fold + test-part noise.
# Pre-specified: a strip is "too noisy" when its true SNR is below 0 dB (Step 3.4: below 0 dB the noise-trained model
# loses more than 0.02 AUC); each gate's threshold maximises sensitivity + specificity for that on validation.
# GPU optional: about 3 min with a T4, about 15 min on CPU.
# ---- 1) Setup ----
import os, sys, time, importlib
import numpy as np, pandas as pd, torch, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
P3 = ROOT
NST = f"{P3}/data/nstdb"
SRC, PKG = f"{P3}/src", f"{P3}/src/leadsuite"

sys.path_importer_cache.pop(SRC, None); importlib.invalidate_caches()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
for m in [k for k in sys.modules if k.startswith("leadsuite")]:
    del sys.modules[m]
from leadsuite.data import SUPER, label_matrix, load_ptbxl, task_mask
from leadsuite.metrics import auc_per_class
from leadsuite.cnn import load_members, predict, preprocess
from leadsuite.noise import ALL, LABELS, add_noise, load_nstdb, noise_bank
from leadsuite.quality import CAP_DB, FEATURES, estimate_snr, sqi_features

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))
BAD_DB, TEST_SNR, TRAIN_SNR = 0.0, (-12.0, 24.0), (-12.0, 30.0)
COPIES = 3                                   # noisy copies per validation / test ECG (plus the clean one)
COVER = [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5]
t_all = time.time()
print(f"[1/6] Device: {DEV}")

# ---- 2) Data: lead II (as in Steps 3.3-3.4), the two lead II ensembles, the noise ----
X, gain, leads, meta = load_ptbxl(f"{P3}/data/ptbxl100.npz", f"{P3}/data/ptbxl_meta.csv")
clean_m, ckc = load_members(f"{P3}/models/cnn_leadII.pt")
noisy_m, _ = load_members(f"{P3}/models/cnn_leadII_noise.pt")
scale = np.float32(ckc["scale"][0])
Xf = preprocess(X[:, :, [leads.index("II")]], gain) / scale
del X
task, Y = task_mask(meta), label_matrix(meta)
split = meta["split"].to_numpy()
tr, va, te = (np.where(task & (split == s))[0] for s in ("train", "val", "test"))
noise = load_nstdb(NST)


def make_set(idx, copies, part, seed, snr_range):
    """The clean strips of idx plus `copies` noisy copies each (random type, random SNR uniform in dB).
    Returns strips [m, 1, L], source row (into meta), true SNR (inf = clean), noise type."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(idx, copies)
    bank = noise_bank(noise, part, len(rows), seed=seed + 100)
    t = rng.integers(0, len(ALL), len(rows))
    s = rng.uniform(*snr_range, len(rows))
    nz = np.stack([bank[ALL[k]][i] for i, k in enumerate(t)])
    x = np.concatenate([Xf[idx], add_noise(Xf[rows], nz, s)])
    return (x, np.concatenate([idx, rows]), np.concatenate([np.full(len(idx), np.inf), s]),
            np.concatenate([np.full(len(idx), "none"), np.array(ALL)[t]]))


xs_tr, _, s_tr, _ = make_set(tr, 1, "train", 11, TRAIN_SNR)
xs_va, r_va, s_va, t_va = make_set(va, COPIES, "train", 12, TEST_SNR)
xs_te, r_te, s_te, t_te = make_set(te, COPIES, "test", 13, TEST_SNR)
print(f"[2/6] Strips: estimator training {len(xs_tr):,} (training fold, training-part noise) | validation "
      f"{len(xs_va):,} | test {len(xs_te):,} ({len(te):,} test ECGs: clean + {COPIES} noisy copies at "
      f"{TEST_SNR[0]:.0f} to {TEST_SNR[1]:.0f} dB, test-part noise)")

# ---- 3) Gate 1: estimate the SNR from the strip; threshold chosen on validation ----
t0 = time.time()
F_tr, F_va, F_te = sqi_features(xs_tr), sqi_features(xs_va), sqi_features(xs_te)
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
reg.fit(F_tr, np.minimum(s_tr, CAP_DB))
gate = {"model": reg, "features": FEATURES, "cap_db": CAP_DB, "bad_db": BAD_DB}


def youden(score_good, bad):
    """Threshold on a 'higher = better quality' score that maximises sensitivity + specificity for `bad`."""
    cand = np.unique(np.quantile(score_good, np.linspace(0.01, 0.99, 197)))
    j = [((score_good < c) & bad).sum() / bad.sum() + ((score_good >= c) & ~bad).sum() / (~bad).sum() for c in cand]
    return float(cand[int(np.argmax(j))])


est_va = estimate_snr(dict(gate, threshold=0), xs_va)
gate["threshold"] = youden(est_va, s_va < BAD_DB)
est_te = estimate_snr(gate, xs_te)
noisy_te = np.isfinite(s_te)
rho = spearmanr(est_te, np.minimum(s_te, CAP_DB)).correlation
mae = np.abs(est_te[noisy_te] - s_te[noisy_te]).mean()
print(f"[3/6] SNR estimator: {len(FEATURES)} features ({', '.join(FEATURES)}) | {time.time() - t0:.0f} s")
print(f"   Test: Spearman correlation with the true SNR {rho:.3f}; mean absolute error on noisy strips {mae:.1f} dB ("
      + ", ".join(f"{LABELS[t]} {np.abs(est_te[t_te == t] - s_te[t_te == t]).mean():.1f}" for t in ALL) + ")")
print(f"   Gate threshold (chosen on validation): diagnose only if the estimated SNR >= {gate['threshold']:.1f} dB")
joblib.dump(gate, f"{P3}/models/quality_gate.joblib")

# ---- 4) Predictions of both ensembles on validation and test strips; Gate 2 = disagreement of the 5 networks ----
t0 = time.time()


def per_net(members, x):
    out = []
    for m in members:
        out.append(predict(m.to(DEV), x, DEV))
        m.cpu()
    return np.stack(out)


pn_va, pn_te = per_net(noisy_m, xs_va), per_net(noisy_m, xs_te)
pc_te = per_net(clean_m, xs_te).mean(0)
pe_te = pn_te.mean(0)
agree_va, agree_te = -pn_va.std(0).mean(1), -pn_te.std(0).mean(1)     # higher = networks agree more
thr2 = youden(agree_va, s_va < BAD_DB)
Yte = Y[r_te]
pid = meta["patient_id"].to_numpy()[r_te]
print(f"[4/6] Predictions: {pn_va.shape[0]} networks x {len(xs_va) + len(xs_te):,} strips in {time.time() - t0:.0f} s")

# ---- 5) What each gate does on the test strips ----
acc = {"signal-based (estimated SNR)": est_te >= gate["threshold"], "model-based (network agreement)": agree_te >= thr2}
bins = [("clean", ~noisy_te)] + [(f"{lo:+.0f} to {hi:+.0f} dB", noisy_te & (s_te >= lo) & (s_te < hi))
                                 for lo, hi in ((12, 24), (6, 12), (0, 6), (-6, 0), (-12, -6))]
rows = []
for g, a in acc.items():
    r = {"gate": g}
    r.update({f"rejected, {b}": f"{(~a[m]).mean():.0%}" for b, m in bins})
    bad = s_te < BAD_DB
    r["sensitivity (SNR < 0 dB rejected)"] = f"{(~a[bad]).mean():.0%}"
    r["false rejections (SNR >= 0 dB)"] = f"{(~a[~bad]).mean():.0%}"
    r["coverage"] = f"{a.mean():.0%}"
    rows.append(r)
rej = pd.DataFrame(rows)
rej.to_csv(f"{P3}/results/quality_gate_rejections.csv", index=False)
pd.set_option("display.width", 250)
print("\n[5/6] Share of test strips each gate refuses, by true noise level:")
print(rej.T.to_string(header=False))

rng = np.random.default_rng(0)
u, inv = np.unique(pid, return_inverse=True)
prow = [np.where(inv == k)[0] for k in range(len(u))]
boots = [np.concatenate([prow[j] for j in rng.integers(0, len(u), len(u))]) for _ in range(300)]


def mauc(rows_, P):
    try:
        return auc_per_class(Yte[rows_], P[rows_])["macro"]
    except ValueError:                      # a class missing from a very small subset
        return np.nan


out = []
for model, P in (("noise-trained", pe_te), ("clean-trained", pc_te)):
    allr = np.arange(len(P))
    for g, a in acc.items():
        ka, kr = np.where(a)[0], np.where(~a)[0]
        d = np.array([mauc(b[a[b]], P) - mauc(b, P) for b in boots])
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out.append({"model": model, "gate": g, "all strips": round(mauc(allr, P), 3),
                    "accepted": round(mauc(ka, P), 3), "refused": round(mauc(kr, P), 3),
                    "accepted minus all (95% CI)": f"{np.mean(d):+.3f} ({lo:+.3f} to {hi:+.3f})",
                    "coverage": f"{a.mean():.0%}"})
gain_tab = pd.DataFrame(out)
gain_tab.to_csv(f"{P3}/results/quality_gate_auc.csv", index=False)
ref = auc_per_class(Y[te], pe_te[:len(te)])["macro"]
print(f"\n   Macro AUC on all test strips, on the accepted ones and on the refused ones ({len(boots)} patient-level "
      f"bootstrap resamples for the gain); clean strips only, noise-trained model: {ref:.3f}")
print(gain_tab.to_string(index=False))

rand = np.random.default_rng(1).random(len(pe_te))
order = {"oracle (true SNR)": np.minimum(s_te, 1e3), "signal-based (estimated SNR)": est_te,
         "model-based (network agreement)": agree_te, "random": rand}
curve = []
for g, sc in order.items():
    r = {"gate": g}
    for c in COVER:
        keep = np.argsort(-sc, kind="stable")[: int(round(c * len(sc)))]
        r[f"{c:.0%}"] = round(mauc(keep, pe_te), 3)
    curve.append(r)
curve = pd.DataFrame(curve)
curve.to_csv(f"{P3}/results/quality_coverage_curve.csv", index=False)
print("\n   Noise-trained model: macro AUC when only the best X% of strips (by each gate's score) are diagnosed:")
print(curve.to_string(index=False))

# ---- 6) Figures ----
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
ax = axes[0]
edges = np.arange(-12, 25, 3)
mid = (edges[:-1] + edges[1:]) / 2
for t, col in zip(ALL, ("#0ea5e9", "#f97316", "#22c55e", "#7c3aed")):
    sel = [est_te[(t_te == t) & (s_te >= a) & (s_te < b)] for a, b in zip(edges[:-1], edges[1:])]
    med = [np.median(v) if len(v) else np.nan for v in sel]
    ax.plot(mid, med, marker="o", ms=3.5, color=col, label=LABELS[t])
ax.plot([-12, 24], [-12, 24], color="#111", lw=0.8, ls=":", label="perfect estimate")
ax.axhline(gate["threshold"], color="#e11d48", ls="--", lw=1, label=f"gate threshold {gate['threshold']:.1f} dB")
ax.set_xlabel("true SNR of the test strip (dB)")
ax.set_ylabel("estimated SNR (median, dB)")
ax.set_title("How well the strip alone tells its own noise level", loc="left", fontsize=10.5)
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
for (_, r), col, ls in zip(curve.iterrows(), ("#111", "#e11d48", "#2563eb", "#9ca3af"), (":", "-", "-", "--")):
    ax.plot([c * 100 for c in COVER], [r[f"{c:.0%}"] for c in COVER], color=col, ls=ls, marker="o", ms=3.5,
            label=r["gate"])
ax.axhline(ref, color="#16a34a", lw=0.9, ls="--", label=f"clean strips only ({ref:.3f})")
ax.invert_xaxis()
ax.set_xlabel("strips diagnosed (%)  →  more refused")
ax.set_ylabel("macro ROC AUC on the diagnosed strips")
ax.set_title("Refusing the noisiest strips (noise-trained lead II ensemble)", loc="left", fontsize=10.5)
ax.legend(frameon=False, fontsize=8, loc="best")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{P3}/results/quality_gate.png", dpi=150)
plt.show()
np.savez_compressed(f"{P3}/results/quality_test_strips.npz", ecg_id=meta.index.to_numpy()[r_te], true_snr=s_te,
                    noise=t_te, est_snr=est_te, agreement=agree_te, p_noise_trained=pe_te, p_clean_trained=pc_te)
print(f"[6/6] Saved: models/quality_gate.joblib, results/quality_*.csv, quality_gate.png, quality_test_strips.npz | "
      f"total {(time.time() - t_all) / 60:.0f} min")
