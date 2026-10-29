# STEP 3.3c — Two design questions for the device, answered on PTB-XL before buying hardware:
#   A) One channel or two? CNNs on lead pairs (I + II, II + V1, II + V2, II + V5) vs lead II alone and vs 12 leads.
#   B) Can lead II name the specific diagnosis? A lead II CNN trained on the 23 PTB-XL diagnostic subclasses
#      (e.g. "inferior MI" instead of just "MI").
# Same recipe as Step 3.3 (clean training, epoch chosen on fold 9, locked test fold 10). 3 seeds per model; the
# references (lead II, lead I, 12-lead) use seeds 0-2 of Step 3.3, so every ensemble has 3 members.
# GPU recommended (about 30-35 min). Each seed is saved, so a re-run resumes.
# ---- 1) Setup ----
import os, sys, time, importlib
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
P3 = ROOT
SRC, PKG = f"{P3}/src", f"{P3}/src/leadsuite"
os.makedirs(f"{P3}/models/cnn_seeds", exist_ok=True)

sys.path_importer_cache.pop(SRC, None); importlib.invalidate_caches()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
for m in [k for k in sys.modules if k.startswith("leadsuite")]:
    del sys.modules[m]
from leadsuite.data import NAMES, SUPER, label_matrix, load_ptbxl, task_mask
from leadsuite.metrics import auc_per_class, macro_auc, patient_bootstrap
from leadsuite.cnn import load_members, predict, preprocess, save_members, train

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS, EPOCHS = [0, 1, 2], 30
REF = {"lead II": "leadII", "lead I": "leadI", "12-lead": "12lead"}
PAIRS = {"I + II": ["I", "II"], "II + V1": ["II", "V1"], "II + V2": ["II", "V2"], "II + V5": ["II", "V5"]}
SUB = {"NORM": {"NORM": "normal ECG"},
       "MI": {"IMI": "inferior MI", "AMI": "anterior MI", "LMI": "lateral MI", "PMI": "posterior MI"},
       "STTC": {"STTC": "ST/T change", "NST_": "non-specific ST change", "ISC_": "non-specific ischaemia",
                "ISCA": "ischaemia, anterior leads", "ISCI": "ischaemia, inferior leads"},
       "CD": {"LAFB/LPFB": "left fascicular block", "IRBBB": "incomplete RBBB", "_AVB": "AV block",
              "IVCD": "non-specific intraventricular block", "CRBBB": "complete RBBB", "CLBBB": "complete LBBB",
              "ILBBB": "incomplete LBBB", "WPW": "Wolff-Parkinson-White"},
       "HYP": {"LVH": "left ventricular hypertrophy", "LAO/LAE": "left atrial enlargement",
               "RAO/RAE": "right atrial enlargement", "RVH": "right ventricular hypertrophy",
               "SEHYP": "septal hypertrophy"}}
SUB23 = [s for c in SUPER for s in SUB[c]]
MIN_POS = 20
print(f"[1/6] Device: {DEV}" + ("" if DEV == "cuda" else "  ⚠️ no GPU: training will take many hours"))

# ---- 2) Data (same preprocessing and per-lead scale as Step 3.3) ----
X, gain, leads, meta = load_ptbxl(f"{P3}/data/ptbxl100.npz", f"{P3}/data/ptbxl_meta.csv")
_, ck12 = load_members(f"{P3}/models/cnn_seeds/12lead_seed0.pt")
Xf = preprocess(X, gain) / np.array(ck12["scale"], np.float32)[None, :, None]
del X
task, Y = task_mask(meta), label_matrix(meta)
split = meta["split"].to_numpy()
tr, va, te = (np.where(task & (split == s))[0] for s in ("train", "val", "test"))
subs = meta["subclass"].fillna("").apply(lambda s: set(x for x in s.split(";") if x))
Y23 = np.array([[int(s in z) for s in SUB23] for z in subs])
assert ((Y23.sum(1) > 0) == task).all(), "every task ECG should have at least one subclass"
pid = meta["patient_id"].to_numpy()[te]
print(f"[2/6] {len(tr):,} train | {len(va):,} val | {len(te):,} test ECGs | 23 subclasses, test ECGs per subclass: "
      + ", ".join(f"{s} {int(Y23[te, j].sum())}" for j, s in enumerate(SUB23)))


def fit(name, li, Ytask, tag):
    """Train (or load) SEEDS for one model; returns the list of networks."""
    Xc = np.ascontiguousarray(Xf[:, li, :])
    out = []
    for s in SEEDS:
        path = f"{P3}/models/cnn_seeds/{tag}_seed{s}.pt"
        if os.path.exists(path):
            m, c = load_members(path)
            out.append(m[0])
            print(f"   {name} seed {s}: loaded (best epoch {c['best_epoch']}, validation {c['best_val']:.3f})")
            continue
        t0 = time.time()
        print(f"   {name} seed {s} ", end="", flush=True)
        model, best, ep = train(Xc, Ytask, tr, va, epochs=EPOCHS, seed=s, device=DEV,
                                log=lambda msg: print(".", end="", flush=True))
        save_members(path, [model], {"seed": s, "best_epoch": ep, "best_val": best, "leads": li,
                                     "scale": [ck12["scale"][i] for i in li]})
        out.append(model)
        print(f" best epoch {ep}, validation macro AUC {best:.3f} | {(time.time() - t0) / 60:.1f} min")
    save_members(f"{P3}/models/cnn_{tag}.pt", out, {"seeds": SEEDS, "epochs": EPOCHS, "leads": li,
                                                    "scale": [ck12["scale"][i] for i in li]})
    return out, Xc


def ens(members, Xc):
    p = []
    for m in members:
        p.append(predict(m.to(DEV), Xc[te], DEV))
        m.cpu()
    return np.mean(p, 0), [auc_per_class(Y[te], q)["macro"] if q.shape[1] == 5 else macro_auc(Y23[te], q) for q in p]


# ---- 3) A) Lead pairs: train 3 seeds each; references = seeds 0-2 of Step 3.3 ----
t_all = time.time()
P, single = {}, {}
for name, tag in REF.items():
    mem = [load_members(f"{P3}/models/cnn_seeds/{tag}_seed{s}.pt")[0][0] for s in SEEDS]
    li = list(range(12)) if name == "12-lead" else [leads.index(name.split()[1])]
    P[name], single[name] = ens(mem, np.ascontiguousarray(Xf[:, li, :]))
for name, pair in PAIRS.items():
    li = [leads.index(l) for l in pair]
    mem, Xc = fit(name, li, Y, "pair_" + "_".join(pair))
    P[name], single[name] = ens(mem, Xc)
order = ["lead I", "lead II"] + list(PAIRS) + ["12-lead"]
rows = []
for k in order:
    a = auc_per_class(Y[te], P[k])
    r = {"input": k, "channels": 12 if k == "12-lead" else (2 if k in PAIRS else 1)}
    r.update({c: round(a[c], 3) for c in SUPER})
    r["macro"] = round(a["macro"], 3)
    r["single seed (mean ± SD)"] = f"{np.mean(single[k]):.3f} ± {np.std(single[k], ddof=1):.3f}"
    rows.append(r)
tabA = pd.DataFrame(rows)
tabA.to_csv(f"{P3}/results/pairs_auc.csv", index=False)
print(f"\n[3/6] A) Test fold ({len(te):,} ECGs), ROC AUC of {len(SEEDS)}-seed ensembles:")
print(tabA.to_string(index=False))
draws = patient_bootstrap(Y[te], {k: P[k] for k in order}, pid, n=1000, seed=0)
cols = SUPER + ["macro"]
diff = []
for k in list(PAIRS) + ["12-lead"]:
    for ref in (["lead II", "12-lead"] if k != "12-lead" else ["lead II"]):
        d = draws[k] - draws[ref]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        r = {"comparison": f"{k} minus {ref}"}
        r.update({c: f"{d[:, j].mean():+.3f} ({lo[j]:+.3f} to {hi[j]:+.3f})" for j, c in enumerate(cols)})
        diff.append(r)
diff = pd.DataFrame(diff)
diff.to_csv(f"{P3}/results/pairs_auc_diff.csv", index=False)
print(f"\n   Paired differences ({len(draws['lead II'])} patient-level bootstrap resamples; interval excluding 0 = real):")
print(diff.to_string(index=False))

srows = []
for c in SUPER[1:]:
    j = SUPER.index(c)
    neg = (meta[c].to_numpy() == 0)[te]
    for s, desc in SUB[c].items():
        pos = ((meta[c].to_numpy() == 1)[te] & np.array([(z & set(SUB[c])) == {s} for z in subs.iloc[te]]))
        if pos.sum() < MIN_POS:
            continue
        keep = pos | neg
        r = {"subclass": s, "meaning": desc, "n": int(pos.sum())}
        r.update({k: round(roc_auc_score(pos[keep].astype(int), P[k][keep, j]), 3) for k in order})
        srows.append(r)
tabS = pd.DataFrame(srows)
tabS.to_csv(f"{P3}/results/pairs_subclass_auc.csv", index=False)
print("\n   Per subclass (same definition as Step 3.3b: that subclass only vs ECGs without its superclass):")
print(tabS.to_string(index=False))

# ---- 4) B) Lead II CNN trained on the 23 diagnostic subclasses ----
mem23, Xc = fit("lead II, 23 subclasses", [leads.index("II")], Y23, "leadII_sub23")
P23, single23 = ens(mem23, Xc)
rng = np.random.default_rng(0)
u, inv = np.unique(pid, return_inverse=True)
prow = [np.where(inv == k)[0] for k in range(len(u))]
boots = [np.concatenate([prow[j] for j in rng.integers(0, len(u), len(u))]) for _ in range(1000)]
rowsB = []
for j, s in enumerate(SUB23):
    y = Y23[te, j]
    if y.sum() < MIN_POS:
        continue
    a = roc_auc_score(y, P23[:, j])
    bs = [roc_auc_score(y[b], P23[b, j]) for b in boots if 0 < y[b].sum() < len(b)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    c = next(k for k in SUPER if s in SUB[k])
    rowsB.append({"superclass": c, "subclass": s, "meaning": SUB[c][s], "test ECGs": int(y.sum()),
                  "AUC": round(a, 3), "95% CI": f"{lo:.3f}-{hi:.3f}"})
tabB = pd.DataFrame(rowsB)
tabB.to_csv(f"{P3}/results/sub23_leadII_auc.csv", index=False)
print(f"\n[4/6] B) Lead II, 23-subclass CNN ({len(SEEDS)}-seed ensemble), one-vs-rest ROC AUC per subclass on the test "
      f"fold (subclasses with >= {MIN_POS} test ECGs; positives: ECGs with that subclass, negatives: all other ECGs):")
print(tabB.to_string(index=False))
print(f"   Macro AUC over these {len(tabB)} subclasses: {tabB['AUC'].mean():.3f} | single seed: "
      f"{np.mean(single23):.3f} ± {np.std(single23, ddof=1):.3f} (over all subclasses present in the test fold)")
iI, iA = SUB23.index("IMI"), SUB23.index("AMI")
mi = [z & set(SUB["MI"]) for z in subs.iloc[te]]
only_i = np.array([z == {"IMI"} for z in mi]); only_a = np.array([z == {"AMI"} for z in mi])
k = only_i | only_a
naming = roc_auc_score(only_i[k].astype(int), (P23[:, iI] - P23[:, iA])[k])
print(f"   Naming the MI: among MI ECGs labelled only inferior ({only_i.sum()}) or only anterior ({only_a.sum()}), "
      f"AUC for telling which = {naming:.3f} (0.5 = cannot tell)")

# ---- 5) Figure ----
fig, ax = plt.subplots(figsize=(10, 4))
lo, hi = np.percentile(np.stack([draws[k][:, -1] for k in order], 1), [2.5, 97.5], axis=0)
y = [auc_per_class(Y[te], P[k])["macro"] for k in order]
colors = ["#9ca3af", "#e11d48"] + ["#f59e0b"] * len(PAIRS) + ["#1f3864"]
ax.bar(range(len(order)), y, color=colors, width=0.6)
ax.errorbar(range(len(order)), y, yerr=[np.array(y) - lo, hi - np.array(y)], fmt="none", ecolor="#111", lw=1,
            capsize=3)
for i, v in enumerate(y):
    ax.text(i, hi[i] + 0.004, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(order)), order)
ax.set_ylim(min(0.75, lo.min() - 0.02), max(0.96, hi.max() + 0.02))
ax.set_ylabel("macro ROC AUC (test fold)")
ax.set_title("One channel, two channels or twelve: what each input gives (3-seed CNN ensembles, 95% CI)", loc="left",
             fontsize=11)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{P3}/results/pairs_macro_auc.png", dpi=150)
plt.show()
print("[5/6] Figure saved: results/pairs_macro_auc.png")

# ---- 6) Save predictions ----
np.savez_compressed(f"{P3}/results/pairs_sub23_probs_test.npz", ecg_id=meta.index.to_numpy()[te],
                    sub23=P23, sub23_names=np.array(SUB23), **{k.replace(" ", ""): v for k, v in P.items()})
print(f"[6/6] Saved: results/pairs_*.csv, sub23_leadII_auc.csv, pairs_macro_auc.png, pairs_sub23_probs_test.npz; "
      f"models/cnn_pair_*.pt, models/cnn_leadII_sub23.pt | total {(time.time() - t_all) / 60:.0f} min")
