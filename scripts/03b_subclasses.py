# STEP 3.3b — Which specific diagnoses survive in one lead? The Step 3.3 CNN ensembles (12-lead, lead II, lead I) on
# the same locked test fold, broken down by PTB-XL diagnostic subclass (e.g. inferior vs anterior MI, AV block).
# No training, no GPU: it only re-reads the saved test predictions (about 1-2 min on CPU).
# Hypotheses written down before looking (from ECG physiology, not from our data):
#   H1 inferior MI (IMI) is detected better from lead II than from lead I (lead II looks at the inferior wall);
#   H2 anterior MI (AMI) is detected poorly from both single leads (it shows mainly in V1-V4);
#   H3 AV block (_AVB) is detected better from lead II (P waves and the PR interval are clearest in lead II).
# Method: for each subclass, positives = test ECGs whose only subclass within that superclass is this one (e.g. MI
# ECGs labelled IMI and no other MI subclass); negatives = test ECGs without that superclass at all. Score = the
# network's output for the superclass. Subclasses with fewer than 20 such positives in the test fold are skipped.
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

P3 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
SUB = {"MI": {"IMI": "inferior MI", "AMI": "anterior MI", "LMI": "lateral MI", "PMI": "posterior MI"},
       "STTC": {"STTC": "ST/T change", "NST_": "non-specific ST change", "ISC_": "non-specific ischaemia",
                "ISCA": "ischaemia, anterior leads", "ISCI": "ischaemia, inferior leads"},
       "CD": {"LAFB/LPFB": "left fascicular block", "IRBBB": "incomplete RBBB", "_AVB": "AV block",
              "IVCD": "non-specific intraventricular block", "CRBBB": "complete RBBB", "CLBBB": "complete LBBB",
              "ILBBB": "incomplete LBBB", "WPW": "Wolff-Parkinson-White"},
       "HYP": {"LVH": "left ventricular hypertrophy", "LAO/LAE": "left atrial enlargement",
               "RAO/RAE": "right atrial enlargement", "RVH": "right ventricular hypertrophy",
               "SEHYP": "septal hypertrophy"}}
MIN_POS, N_BOOT = 20, 1000
CFG = {"12-lead": "p12", "lead II": "pII", "lead I": "pI"}

meta = pd.read_csv(f"{P3}/data/ptbxl_meta.csv", index_col=0)
r = np.load(f"{P3}/results/cnn_probs_test.npz")
m = meta.loc[r["ecg_id"]]
subs = m["subclass"].fillna("").apply(lambda s: set(x for x in s.split(";") if x))
pid = m["patient_id"].to_numpy()
print(f"[1/3] Test fold: {len(m):,} ECGs from {len(np.unique(pid)):,} patients | predictions of the Step 3.3 ensembles")


def boot(y, scores, pid, n=N_BOOT, seed=0):
    """Patient-level bootstrap of the AUC (all ECGs of a patient are resampled together)."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(pid, return_inverse=True)
    rows = [np.where(inv == k)[0] for k in range(len(u))]
    out = {k: [] for k in scores}
    for _ in range(n):
        idx = np.concatenate([rows[j] for j in rng.integers(0, len(u), len(u))])
        if y[idx].min() == y[idx].max():
            continue
        for k, s in scores.items():
            out[k].append(roc_auc_score(y[idx], s[idx]))
    return {k: np.array(v) for k, v in out.items()}


def fmt(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{d.mean():+.3f} ({lo:+.3f} to {hi:+.3f})" + (" *" if lo > 0 or hi < 0 else "")


rows = []
for c, names in SUB.items():
    j = SUPER.index(c)
    neg = (m[c] == 0).to_numpy()
    for s, desc in names.items():
        pos = ((m[c] == 1) & subs.apply(lambda z, s=s: (z & set(names)) == {s})).to_numpy()
        if pos.sum() < MIN_POS:
            print(f"   skipped {s} ({desc}): only {int(pos.sum())} test ECGs")
            continue
        keep = pos | neg
        y, ids = pos[keep].astype(int), pid[keep]
        sc = {k: r[v][keep, j] for k, v in CFG.items()}
        d = boot(y, sc, ids)
        row = {"superclass": c, "subclass": s, "meaning": desc, "test ECGs": int(pos.sum())}
        row.update({f"AUC {k}": round(roc_auc_score(y, v), 3) for k, v in sc.items()})
        row["lead II - lead I"] = fmt(d["lead II"] - d["lead I"])
        row["lead II - 12-lead"] = fmt(d["lead II"] - d["12-lead"])
        rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(f"{P3}/results/subclass_auc.csv", index=False)
print(f"\n[2/3] AUC per subclass (positives: that subclass only, within its superclass; negatives: no such superclass). "
      f"Differences: mean and 95% interval from {N_BOOT} patient-level bootstrap resamples; * = interval excludes 0.")
pd.set_option("display.width", 250)
print(tab.to_string(index=False))
print(f"   Note: {len(tab)} subclasses x 2 comparisons. With this many intervals, one or two may exclude 0 by chance;\n"
      "   only H1-H3 were stated in advance, everything else is exploratory.")

fig, ax = plt.subplots(figsize=(9, 0.42 * len(tab) + 1.4))
yy = np.arange(len(tab))[::-1]
for k, col, mk in (("12-lead", "#1f3864", "o"), ("lead II", "#e11d48", "s"), ("lead I", "#9ca3af", "D")):
    ax.scatter(tab[f"AUC {k}"], yy, color=col, marker=mk, s=34, label=k, zorder=3)
for y_, (_, t) in zip(yy, tab.iterrows()):
    ax.plot([min(t["AUC lead I"], t["AUC lead II"]), max(t["AUC lead I"], t["AUC lead II"])], [y_, y_], color="#ddd",
            lw=3, zorder=1)
ax.set_yticks(yy, [f"{t['meaning']} ({t['subclass']}, n={t['test ECGs']})" for _, t in tab.iterrows()], fontsize=8.5)
ax.set_xlabel("ROC AUC on the test fold (that subclass vs ECGs without its superclass)")
ax.set_xlim(min(0.5, tab[[f"AUC {k}" for k in CFG]].min().min() - 0.03), 1.0)
ax.grid(axis="x", alpha=0.3)
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
fig.tight_layout()
fig.savefig(f"{P3}/results/subclass_auc.png", dpi=150)
plt.show()
print("[3/3] Saved: results/subclass_auc.csv, results/subclass_auc.png")
