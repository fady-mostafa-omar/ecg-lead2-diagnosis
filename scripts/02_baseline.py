# STEP 3.2 — Baseline: hand-crafted features (beat timing + the median beat of each lead) and gradient boosting,
# for 12 leads, lead II alone and lead I alone. Scored once on the locked test fold (fold 10). CPU is fine (~15 min).
# ---- 1) Setup: write the leadsuite package to Drive ----
import os, sys, time, importlib
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
P3 = ROOT
SRC, PKG = f"{P3}/src", f"{P3}/src/leadsuite"
for d in (PKG, f"{P3}/models", f"{P3}/results"):
    os.makedirs(d, exist_ok=True)

sys.path_importer_cache.pop(SRC, None); importlib.invalidate_caches()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
for m in [k for k in sys.modules if k.startswith("leadsuite")]:
    del sys.modules[m]
from leadsuite.data import FS, NAMES, SUPER, label_matrix, load_ptbxl, task_mask
from leadsuite.features import RR_KEYS, feature_names, features_one
from leadsuite.metrics import auc_per_class, patient_bootstrap
print("[1/5] leadsuite package:", PKG)

# ---- 2) Features (cached on Drive): pass A = R peaks on lead II, all 12 leads; pass B = R peaks on lead I ----
X, gain, leads, meta = load_ptbxl(f"{P3}/data/ptbxl100.npz", f"{P3}/data/ptbxl_meta.csv")
iI, iII = leads.index("I"), leads.index("II")


def compute(path, detect, use):
    if os.path.exists(path):
        return np.load(path)["F"]
    t0 = time.time()
    jobs = ((X[i].astype(np.float32) / gain, FS, detect, use) for i in range(len(X)))
    out = []
    with Pool(os.cpu_count()) as pool:
        for i, f in enumerate(pool.imap(features_one, jobs, chunksize=50), 1):
            out.append(f)
            if i % 5000 == 0:
                print(f"   {os.path.basename(path)}: {i:,}/{len(X):,} ECGs ({time.time() - t0:.0f} s)", flush=True)
    F = np.stack(out)
    np.savez_compressed(path, F=F)
    return F


FA = compute(f"{P3}/data/features_detII.npz", iII, list(range(12)))
FB = compute(f"{P3}/data/features_detI.npz", iI, [iI])
names_A, names_B = feature_names(leads), feature_names(["I"])
assert FA.shape[1] == len(names_A) and FB.shape[1] == len(names_B)
colsII = [k for k, n in enumerate(names_A) if n in RR_KEYS or n.startswith("II_")]
CONFIG = {"12-lead": FA, "lead II": FA[:, colsII], "lead I": FB}
nb_A, nb_B = FA[:, names_A.index("n_beats")], FB[:, names_B.index("n_beats")]
print(f"[2/5] Features: 12-lead {FA.shape[1]}, lead II {len(colsII)}, lead I {FB.shape[1]} per ECG | beats per 10 s "
      f"(median): lead II {np.median(nb_A):.0f}, lead I {np.median(nb_B):.0f} | ECGs with fewer than 3 beats found: "
      f"lead II {int((nb_A < 3).sum())}, lead I {int((nb_B < 3).sum())}")

# ---- 3) Gradient boosting, one model per superclass; fixed settings (no tuning); fold 9 is only reported ----
task, Y = task_mask(meta), label_matrix(meta)
split = meta["split"].to_numpy()
tr, va, te = (task & (split == s) for s in ("train", "val", "test"))
probs, rows, models = {}, [], {}
for cfg, F in CONFIG.items():
    t0 = time.time()
    P = np.zeros((len(F), len(SUPER)))
    models[cfg] = []
    for k, c in enumerate(SUPER):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=0)
        m.fit(F[tr], Y[tr, k])
        P[:, k] = m.predict_proba(F)[:, 1]
        models[cfg].append(m)
    probs[cfg] = P
    for s, mask in (("val (fold 9)", va), ("test (fold 10)", te)):
        a = auc_per_class(Y[mask], P[mask])
        row = {"split": s, "config": cfg}
        row.update({c: round(a[c], 3) for c in SUPER})
        row["macro AUC"] = round(a["macro"], 3)
        rows.append(row)
    print(f"   {cfg}: trained in {time.time() - t0:.0f} s", flush=True)
auc = pd.DataFrame(rows)
print("\n[3/5] ROC AUC per superclass (1.0 = perfect, 0.5 = chance):")
print(auc.to_string(index=False))
print("   Reported for context (Strodthoff et al., 12-lead deep networks, same test fold): macro AUC 0.921-0.930; "
      "Wavelet features + neural network 0.874")

# ---- 4) Patient-level bootstrap on the test fold: confidence intervals and what lead II loses ----
Yt, pt = Y[te], meta["patient_id"].to_numpy()[te]
draws = patient_bootstrap(Yt, {k: v[te] for k, v in probs.items()}, pt, n=1000, seed=0)
cols = SUPER + ["macro"]
ci = []
for cfg in CONFIG:
    lo, hi = np.percentile(draws[cfg], [2.5, 97.5], axis=0)
    row = {"config": cfg}
    row.update({c: f"{lo[j]:.3f}-{hi[j]:.3f}" for j, c in enumerate(cols)})
    ci.append(row)
diff = []
for a_, b_ in (("lead II", "12-lead"), ("lead II", "lead I")):
    d = draws[a_] - draws[b_]
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    row = {"comparison": f"{a_} minus {b_}"}
    row.update({c: f"{d[:, j].mean():+.3f} ({lo[j]:+.3f} to {hi[j]:+.3f})" for j, c in enumerate(cols)})
    diff.append(row)
print(f"\n[4/5] 95% confidence intervals (test fold, {len(draws['12-lead'])} patient-level bootstrap resamples):")
print(pd.DataFrame(ci).to_string(index=False))
print("\n   Paired differences in AUC (mean and 95% interval; an interval that excludes 0 is a real difference):")
print(pd.DataFrame(diff).to_string(index=False))

# ---- 5) Save, and one figure ----
auc.to_csv(f"{P3}/results/baseline_auc.csv", index=False)
pd.DataFrame(ci).to_csv(f"{P3}/results/baseline_auc_ci.csv", index=False)
pd.DataFrame(diff).to_csv(f"{P3}/results/baseline_auc_diff.csv", index=False)
np.savez_compressed(f"{P3}/results/baseline_probs.npz", ecg_id=meta.index.to_numpy(), p12=probs["12-lead"],
                    pII=probs["lead II"], pI=probs["lead I"])
joblib.dump({"models": models, "columns_lead2": colsII, "names_A": names_A, "names_B": names_B},
            f"{P3}/models/baseline_hgb.joblib")
fig, ax = plt.subplots(figsize=(10, 4.2))
colors = {"12-lead": "#1f3864", "lead II": "#e11d48", "lead I": "#9ca3af"}
for j, cfg in enumerate(CONFIG):
    a = auc[(auc["config"] == cfg) & (auc["split"] == "test (fold 10)")].iloc[0]
    lo, hi = np.percentile(draws[cfg], [2.5, 97.5], axis=0)
    x = np.arange(len(cols)) + (j - 1) * 0.26
    y = [a[c] for c in SUPER] + [a["macro AUC"]]
    ax.bar(x, y, 0.26, color=colors[cfg], label=cfg)
    ax.errorbar(x, y, yerr=[np.array(y) - lo, hi - np.array(y)], fmt="none", ecolor="#111", lw=0.8, capsize=2)
ax.set_xticks(range(len(cols)), [NAMES[c] for c in SUPER] + ["Macro AUC"], fontsize=8.5)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ROC AUC (test fold)")
ax.set_title("Baseline (features + gradient boosting): what each lead configuration detects", fontsize=11, loc="left")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{P3}/results/baseline_auc_by_class.png", dpi=150)
plt.show()
print("[5/5] Saved results/baseline_auc*.csv, results/baseline_probs.npz, models/baseline_hgb.joblib and the figure")
