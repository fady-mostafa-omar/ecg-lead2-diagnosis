# STEP 3.6c — External test on ALL 45,053 Chapman-Shaoxing ECGs, with two label mappings reported side by side:
#   "pre-specified": exactly the codes fixed at the top of Step 3.6, before any result was seen;
#   "PTB-XL-consistent" (post hoc, decided after the 10,000-ECG run and the code inventory of Step 3.6b, using only
#   PTB-XL's own statement table and the official PhysioNet Challenge 2021 code names, not any AUC):
#     a Chapman code is a POSITIVE for a superclass when its PTB-XL counterpart is a diagnostic statement of that
#     superclass; it is AMBIGUOUS (left out of that task, neither positive nor negative) when its PTB-XL counterpart is
#     only a "form" statement (e.g. T-wave inversion, ST depression, high QRS voltage, prolonged PR, Q waves);
#     paced ECGs (10370003, rhythm-only in PTB-XL, QRS distorted by the pacemaker) are left out of every task.
#     This adds codes missing from ConditionNames_SNOMED-CT.csv: CRBBB 713427006, IRBBB 713426002, CLBBB 733534002,
#     LAnFB 445118002, BBB 6374002, ST changes 55930002, anterior MI 54329005, acute MI 57054005, ...
# No retraining. Reads data/chapman100.npz saved by Step 3.6 (no download). GPU needed (about 10 min); on CPU it
# would take more than 3 hours, so it stops unless FORCE_CPU = True.
# ---- 1) Setup ----
import os, sys, json, time, importlib
import numpy as np, pandas as pd, torch, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

FORCE_CPU = os.environ.get("FORCE_CPU") == "1"
P3 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = f"{P3}/src"
sys.path_importer_cache.pop(SRC, None); importlib.invalidate_caches()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
for m in [k for k in sys.modules if k.startswith("leadsuite")]:
    del sys.modules[m]
from leadsuite.data import SUPER, label_matrix
from leadsuite.cnn import load_members, predict, preprocess
from leadsuite.quality import estimate_snr

DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda" or FORCE_CPU, "No GPU: Runtime -> Change runtime type -> T4 GPU (or run with FORCE_CPU=1: 3+ hours)"
SR, PACED, N_BOOT = 426783006, 10370003, 500
MAPS = {
    "pre-specified": {
        "MI": ({164865005}, set()), "STTC": ({428750005, 429622005, 164934002, 59931005}, set()),
        "CD": ({59118001, 164909002, 270492004, 195042002, 54016002, 28189009, 27885002, 698252002, 74390002}, set()),
        "HYP": ({164873001, 89792004, 446358003}, set()),
        "RBBB -> CRBBB": ({59118001}, set()), "1AVB -> _AVB": ({270492004}, set()), "LVH -> LVH": ({164873001}, set())},
    "PTB-XL-consistent": {
        "MI": ({164865005, 54329005, 57054005}, {164917005}),
        "STTC": ({428750005, 55930002, 111975006}, {164934002, 59931005, 429622005, 164930006, 164931005}),
        "CD": ({59118001, 713427006, 713426002, 164909002, 733534002, 251120003, 6374002, 445118002, 445211001,
                270492004, 195042002, 54016002, 28189009, 426183003, 27885002, 233917008, 698252002, 74390002,
                195060002}, {164947007}),
        "HYP": ({164873001, 89792004, 446358003, 446813000, 67741000119109}, {55827005, 67751000119106, 251223006}),
        "RBBB -> CRBBB": ({59118001, 713427006}, {713426002}), "1AVB -> _AVB": ({270492004}, {164947007}),
        "LVH -> LVH": ({164873001}, {55827005})}}
NAMED = {"RBBB -> CRBBB": "CRBBB", "1AVB -> _AVB": "_AVB", "LVH -> LVH": "LVH"}
json.dump({k: {t: {"positive": sorted(p), "ambiguous": sorted(a)} for t, (p, a) in v.items()} for k, v in MAPS.items()},
          open(f"{P3}/results/external_chapman_mapping.json", "w"), indent=1)
t_all = time.time()
print(f"[1/5] Device: {DEV}")

# ---- 2) Data and labels under both mappings ----
d = np.load(f"{P3}/data/chapman100.npz")
cm = pd.read_csv(f"{P3}/data/chapman_meta.csv")
assert (cm["record"].astype(str).to_numpy() == d["record"]).all()
codes = cm["dx"].fillna("").apply(lambda s: {int(c) for c in str(s).split(",") if c.strip().isdigit()})
paced = codes.apply(lambda z: PACED in z).to_numpy()
lab = {}
for mp, tasks in MAPS.items():
    excl = paced if mp == "PTB-XL-consistent" else np.zeros(len(cm), bool)
    has = {t: codes.apply(lambda z, p=p: bool(z & p)).to_numpy() for t, (p, a) in tasks.items()}
    amb = {t: codes.apply(lambda z, a=a: bool(z & a)).to_numpy() for t, (p, a) in tasks.items()}
    lab[mp] = {}
    pos = codes.apply(lambda z: z == {SR}).to_numpy()
    neg = np.any([has[t] for t in ("MI", "STTC", "CD", "HYP")], 0)
    lab[mp]["NORM"] = (pos, (pos | neg) & ~excl)
    for t in tasks:
        lab[mp][t] = (has[t], (has[t] | ~amb[t]) & ~excl)
print(f"[2/5] {len(cm):,} ECGs ({paced.sum():,} paced) | positives / ECGs in the task:")
for mp in MAPS:
    print(f"   {mp:18s} " + ", ".join(f"{t} {int((p & u).sum()):,}/{int(u.sum()):,}" for t, (p, u) in lab[mp].items()))

# ---- 3) Predictions of the four ensembles + quality gate ----
t0 = time.time()
Xb = preprocess(d["X"], float(d["gain"]))


def ensemble(path):
    mem, ck = load_members(path)
    x = np.ascontiguousarray(Xb[:, ck["leads"], :] / np.array(ck["scale"], np.float32)[None, :, None])
    p = [predict(m.to(DEV), x, DEV) for m in mem]
    return np.mean(p, 0), x


P = {}
P["12-lead"], _ = ensemble(f"{P3}/models/cnn_12lead.pt")
P["lead II, clean-trained"], xII = ensemble(f"{P3}/models/cnn_leadII.pt")
P["lead II, noise-trained"], _ = ensemble(f"{P3}/models/cnn_leadII_noise.pt")
P23, _ = ensemble(f"{P3}/models/cnn_leadII_sub23.pt")
del Xb
sub23 = [str(s) for s in np.load(f"{P3}/results/pairs_sub23_probs_test.npz")["sub23_names"]]
gate = joblib.load(f"{P3}/models/quality_gate.joblib")
est = estimate_snr(gate, xII[:, 0])
ok = est >= gate["threshold"]
print(f"[3/5] Predictions on {len(xII):,} ECGs in {time.time() - t0:.0f} s | quality gate: {(~ok).mean():.1%} of the real "
      f"clinical lead II strips refused (paced: {(~ok[paced]).mean():.1%}; not paced: {(~ok[~paced]).mean():.1%})")

# ---- 4) AUC with 95% CI, both mappings, next to the PTB-XL test fold ----
ptb = pd.read_csv(f"{P3}/data/ptbxl_meta.csv", index_col=0)
rp = np.load(f"{P3}/results/cnn_probs_test.npz")
rn = np.load(f"{P3}/results/noise_probs_test.npz")
rs = np.load(f"{P3}/results/pairs_sub23_probs_test.npz")
Yp = label_matrix(ptb.loc[rp["ecg_id"]])
subs_p = ptb.loc[rs["ecg_id"], "subclass"].fillna("").apply(lambda s: set(s.split(";")))
internal = {"12-lead": rp["p12"], "lead II, clean-trained": rp["pII"],
            "lead II, noise-trained": rn["II|noise-trained|none|None"]}
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(xII), len(xII)) for _ in range(N_BOOT)]


def auc_ci(y, s, use):
    a = roc_auc_score(y[use], s[use])
    bs = [roc_auc_score(y[b[use[b]]], s[b[use[b]]]) for b in boots]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a, lo, hi


rows = []
for mp in MAPS:
    for t, (pos, use) in lab[mp].items():
        y = pos.astype(int)
        if t in SUPER:
            j = SUPER.index(t)
            for model in P:
                a, lo, hi = auc_ci(y, P[model][:, j], use)
                g = roc_auc_score(y[use & ok], P[model][use & ok, j]) if model.endswith("noise-trained") else np.nan
                rows.append({"mapping": mp, "task": t, "model": model, "positives": int((pos & use).sum()),
                             "AUC external": round(a, 3), "95% CI": f"{lo:.3f}-{hi:.3f}",
                             "gate-accepted": "" if np.isnan(g) else round(g, 3),
                             "AUC internal": round(roc_auc_score(Yp[:, j], internal[model][:, j]), 3)})
        else:
            k = sub23.index(NAMED[t])
            a, lo, hi = auc_ci(y, P23[:, k], use)
            yi = subs_p.apply(lambda z, s=NAMED[t]: s in z).to_numpy().astype(int)
            rows.append({"mapping": mp, "task": t, "model": "lead II, 23 subclasses", "positives": int((pos & use).sum()),
                         "AUC external": round(a, 3), "95% CI": f"{lo:.3f}-{hi:.3f}", "gate-accepted": "",
                         "AUC internal": round(roc_auc_score(yi, rs["sub23"][:, k]), 3)})
ext = pd.DataFrame(rows)
ext["external minus internal"] = (ext["AUC external"] - ext["AUC internal"]).round(3)
ext.to_csv(f"{P3}/results/external_chapman_auc_full.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)
for mp in MAPS:
    print(f"\n[4/5] {mp} mapping — all {len(cm):,} Chapman ECGs ({N_BOOT} bootstrap resamples); internal = PTB-XL test fold:")
    print(ext[ext.mapping == mp].drop(columns="mapping").to_string(index=False))
    sup = ext[(ext.mapping == mp) & ext.task.isin(SUPER)]
    print("   Mean over the 5 superclasses: " + " | ".join(
        f"{m}: external {g['AUC external'].mean():.3f} vs internal {g['AUC internal'].mean():.3f}"
        for m, g in sup.groupby("model", sort=False)))

# ---- 5) Figure and save ----
fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharey=True)
col = {"12-lead": "#1f3864", "lead II, clean-trained": "#fda4af", "lead II, noise-trained": "#e11d48",
       "lead II, 23 subclasses": "#f59e0b"}
tasks = list(lab["pre-specified"])
lo_y = min(0.5, ext[["AUC external", "AUC internal"]].min().min() - 0.03)
for ax, mp in zip(axes, MAPS):
    seen = set()
    for i, t in enumerate(tasks):
        g = ext[(ext.mapping == mp) & (ext.task == t)].reset_index(drop=True)
        for k, (_, r) in enumerate(g.iterrows()):
            x0 = i + (k - (len(g) - 1) / 2) * 0.2
            ax.bar(x0, r["AUC external"], 0.2, color=col[r["model"]], label=None if r["model"] in seen else r["model"])
            seen.add(r["model"])
            ax.scatter(x0, r["AUC internal"], marker="_", s=120, color="#111", zorder=3,
                       label="internal (PTB-XL test)" if "int" not in seen else None)
            seen.add("int")
    ax.set_xticks(range(len(tasks)), tasks, fontsize=8.5)
    ax.set_ylim(lo_y, 1.0)
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"Chapman-Shaoxing, all {len(cm):,} ECGs, {mp} mapping (bars) vs PTB-XL test fold (black marks)",
                 loc="left", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
axes[1].legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(f"{P3}/results/external_chapman_auc_full.png", dpi=150)
plt.show()
np.savez_compressed(f"{P3}/results/external_chapman_probs_full.npz", record=cm["record"].astype(str).to_numpy(),
                    est_snr=est, paced=paced, p12=P["12-lead"], pII=P["lead II, clean-trained"],
                    pII_noise=P["lead II, noise-trained"], sub23=P23)
print(f"[5/5] Saved: results/external_chapman_auc_full.csv, external_chapman_mapping.json, external_chapman_auc_full.png,"
      f" external_chapman_probs_full.npz | total {(time.time() - t_all) / 60:.0f} min")
