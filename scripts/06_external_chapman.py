# STEP 3.6 — External test: the PTB-XL models (trained in Germany) on another country's hospital, without any
# retraining: Chapman-Shaoxing / Ningbo "A large scale 12-lead ECG database for arrhythmia study" (PhysioNet
# ecg-arrhythmia 1.0.0, 45,152 10-s ECGs, 500 Hz, China, CC BY 4.0). Its labels are SNOMED-CT codes, so only findings
# with a clean match to PTB-XL are tested (fixed below, before looking at any result):
#   NORM  positives: sinus rhythm and no other code;  negatives: ECGs with any finding of the five tasks below
#   MI    164865005 (any wall)                         STTC 428750005, 429622005, 164934002, 59931005 (ST-T, ST drop, T wave)
#   CD    59118001 RBBB, 164909002 LBBB (Chapman merges LBBB with fascicular blocks), 270492004 1AVB, 195042002 /
#         54016002 / 28189009 / 27885002 2nd-3rd degree AVB, 698252002 IVB, 74390002 WPW
#   HYP   164873001 LVH, 89792004 RVH, 446358003 RAH
#   Named findings for the 23-subclass lead II model: RBBB -> CRBBB, 1AVB -> _AVB, LVH -> LVH.
# For MI ... HYP: positives = ECGs with the code(s), negatives = ECGs without them. Models: 12-lead, lead II (clean-trained
# and noise-trained), lead II 23-subclass. The quality gate of Step 3.5 is also checked on these real clinical strips.
# GPU recommended (about 25 min, mostly download and reading). Without a GPU, a fixed random 10,000 ECGs are used.
# ---- 1) Setup ----
import os, sys, glob, time, subprocess, importlib
import numpy as np, pandas as pd, torch, joblib, wfdb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
P3 = ROOT
SRC = f"{P3}/src"
LOCAL = f"{P3}/data/raw/chapman"            # 90,000 small files
NPZ = f"{P3}/data/chapman100.npz"                  # compact copy on Drive (re-runs skip download and reading)
sys.path_importer_cache.pop(SRC, None); importlib.invalidate_caches()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
for m in [k for k in sys.modules if k.startswith("leadsuite")]:
    del sys.modules[m]
from leadsuite.data import SUPER, label_matrix, load_ptbxl, task_mask
from leadsuite.metrics import auc_per_class
from leadsuite.cnn import load_members, predict, preprocess
from leadsuite.quality import estimate_snr

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PTB_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TASKS = {"NORM": None, "MI": {164865005}, "STTC": {428750005, 429622005, 164934002, 59931005},
         "CD": {59118001, 164909002, 270492004, 195042002, 54016002, 28189009, 27885002, 698252002, 74390002},
         "HYP": {164873001, 89792004, 446358003}}
NAMED = {"RBBB -> CRBBB": (59118001, "CRBBB"), "1AVB -> _AVB": (270492004, "_AVB"), "LVH -> LVH": (164873001, "LVH")}
SR = 426783006
N_BOOT = 500
t_all = time.time()
print(f"[1/6] Device: {DEV}")


# ---- 2) Download and read (once; then the compact copy on Drive is used) ----
def read_one(path):
    """One record -> (name, int16 [1000, 12] in uV at 100 Hz in PTB-XL lead order, codes, age, sex) or None."""
    try:
        rec = wfdb.rdrecord(path[:-4])
        names = [s.upper() for s in rec.sig_name]
        if rec.fs != 500 or rec.sig_len != 5000 or any(l not in names for l in PTB_LEADS):
            return None
        x = rec.p_signal[:, [names.index(l) for l in PTB_LEADS]]
        if not np.isfinite(x).all():
            return None
        if str(rec.units[0]).lower() in ("uv", "µv"):
            x = x / 1000.0
        info = {c.split(":", 1)[0].strip(): c.split(":", 1)[1].strip() for c in rec.comments if ":" in c}
        codes = [int(c) for c in info.get("Dx", "").split(",") if c.strip().isdigit()]
        x100 = np.clip(np.round(resample_poly(x, 1, 5, axis=0) * 1000), -32768, 32767).astype(np.int16)
        return (os.path.basename(path)[:-4], x100, codes,
                info.get("Age", ""), info.get("Sex", ""))
    except Exception:
        return None


if not os.path.exists(NPZ):
    t0 = time.time()
    r = subprocess.run(f"aws s3 sync --no-sign-request --only-show-errors s3://physionet-open/ecg-arrhythmia/1.0.0/ "
                       f"{LOCAL}", shell=True, capture_output=True, text=True)
    heads = sorted(glob.glob(f"{LOCAL}/**/*.hea", recursive=True))
    if len(heads) < 40000:
        print("AWS copy incomplete (", r.stderr[-300:], ") -> downloading the zip from physionet.org (2.3 GB, slower)")
        url = ("https://physionet.org/static/published-projects/ecg-arrhythmia/"
               "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip")
        subprocess.run(f"wget -q -c -O {LOCAL}/c.zip {url} && unzip -q -o {LOCAL}/c.zip -d {LOCAL}/zip", shell=True)
        heads = sorted(glob.glob(f"{LOCAL}/**/*.hea", recursive=True))
    print(f"   downloaded {len(heads):,} headers in {time.time() - t0:.0f} s; reading ...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max(2, os.cpu_count() or 2)) as ex:
        res = list(ex.map(read_one, heads, chunksize=64))
    ok = [r for r in res if r is not None]
    X = np.stack([r[1] for r in ok])
    cm = pd.DataFrame({"record": [r[0] for r in ok], "dx": [",".join(map(str, r[2])) for r in ok],
                       "age": [r[3] for r in ok], "sex": [r[4] for r in ok]})
    np.savez_compressed(NPZ, X=X, gain=1000.0, leads=np.array(PTB_LEADS), record=cm["record"].to_numpy().astype(str))
    cm.to_csv(f"{P3}/data/chapman_meta.csv", index=False)
    print(f"   read {len(ok):,} usable ECGs ({len(heads) - len(ok)} skipped: wrong length, missing lead or NaN) in "
          f"{time.time() - t0:.0f} s")
d = np.load(NPZ)
X, cm = d["X"], pd.read_csv(f"{P3}/data/chapman_meta.csv")
assert (cm["record"].astype(str).to_numpy() == d["record"]).all()
if DEV == "cpu" and len(X) > 10000:
    keep = np.sort(np.random.default_rng(0).choice(len(X), 10000, replace=False))
    X, cm = X[keep], cm.iloc[keep].reset_index(drop=True)
    print("   ⚠️ no GPU: using a fixed random 10,000 ECGs")
codes = cm["dx"].fillna("").apply(lambda s: {int(c) for c in str(s).split(",") if c.strip().isdigit()})
print(f"[2/6] Chapman-Shaoxing: {len(X):,} ECGs at 100 Hz | most common codes: " + ", ".join(
    f"{k} {v:,}" for k, v in pd.Series([c for z in codes for c in z]).value_counts().head(8).items()))

# ---- 3) Labels for the pre-specified tasks ----
any_finding = codes.apply(lambda z: bool(z & set().union(*[v for v in TASKS.values() if v])))
lab = {}
for t, cs in TASKS.items():
    if t == "NORM":
        pos = codes.apply(lambda z: z == {SR}).to_numpy()
        lab[t] = (pos, pos | any_finding.to_numpy())
    else:
        pos = codes.apply(lambda z, cs=cs: bool(z & cs)).to_numpy()
        lab[t] = (pos, np.ones(len(pos), bool))
for name, (code, _) in NAMED.items():
    pos = codes.apply(lambda z, c=code: c in z).to_numpy()
    lab[name] = (pos, np.ones(len(pos), bool))
print("[3/6] Positives per task: " + ", ".join(f"{k} {int(p.sum()):,} / {int(u.sum()):,}" for k, (p, u) in lab.items()))

# ---- 4) Predictions: same preprocessing and the PTB-XL per-lead scale stored with each model ----
t0 = time.time()
Xb = preprocess(X, float(d["gain"]))                       # [n, 12, 1000] mV, band-passed 0.5-40 Hz
del X


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
sub23 = [str(s) for s in np.load(f"{P3}/results/pairs_sub23_probs_test.npz")["sub23_names"]]
gate = joblib.load(f"{P3}/models/quality_gate.joblib")
est = estimate_snr(gate, xII[:, 0])
ok = est >= gate["threshold"]
print(f"[4/6] Predictions of 4 ensembles on {len(xII):,} ECGs in {time.time() - t0:.0f} s | quality gate on these real "
      f"clinical lead II strips: {(~ok).mean():.1%} refused (estimated SNR median {np.median(est):.1f} dB)")

# ---- 5) AUC: external (Chapman) next to internal (PTB-XL test fold) ----
ptb = pd.read_csv(f"{P3}/data/ptbxl_meta.csv", index_col=0)
rp = np.load(f"{P3}/results/cnn_probs_test.npz")
Yp = label_matrix(ptb.loc[rp["ecg_id"]])
rn = np.load(f"{P3}/results/noise_probs_test.npz")
rs = np.load(f"{P3}/results/pairs_sub23_probs_test.npz")
subs_p = ptb.loc[rs["ecg_id"], "subclass"].fillna("").apply(lambda s: set(s.split(";")))
internal = {"12-lead": rp["p12"], "lead II, clean-trained": rp["pII"], "lead II, noise-trained": rn["II|noise-trained|none|None"]}
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(xII), len(xII)) for _ in range(N_BOOT)]


def auc_ci(y, s, use):
    a = roc_auc_score(y[use], s[use])
    bs = []
    for b in boots:
        b = b[use[b]]
        if 0 < y[b].sum() < len(b):
            bs.append(roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a, lo, hi


rows = []
for t in TASKS:
    pos, use = lab[t]
    j = SUPER.index(t)
    for model in P:
        a, lo, hi = auc_ci(pos.astype(int), P[model][:, j], use)
        a_g = roc_auc_score(pos[use & ok], P[model][use & ok, j]) if model.endswith("noise-trained") else np.nan
        rows.append({"task": t, "model": model, "Chapman positives": int(pos.sum()),
                     "AUC external": round(a, 3), "95% CI": f"{lo:.3f}-{hi:.3f}",
                     "AUC external, gate-accepted": round(a_g, 3) if np.isfinite(a_g) else "",
                     "AUC internal (PTB-XL test)": round(roc_auc_score(Yp[:, j], internal[model][:, j]), 3)})
for name, (code, s) in NAMED.items():
    pos, use = lab[name]
    k = sub23.index(s)
    a, lo, hi = auc_ci(pos.astype(int), P23[:, k], use)
    yi = subs_p.apply(lambda z, s=s: s in z).to_numpy().astype(int)
    rows.append({"task": name, "model": "lead II, 23 subclasses", "Chapman positives": int(pos.sum()),
                 "AUC external": round(a, 3), "95% CI": f"{lo:.3f}-{hi:.3f}", "AUC external, gate-accepted": "",
                 "AUC internal (PTB-XL test)": round(roc_auc_score(yi, rs["sub23"][:, k]), 3)})
ext = pd.DataFrame(rows)
ext["external minus internal"] = (ext["AUC external"] - ext["AUC internal (PTB-XL test)"]).round(3)
ext.to_csv(f"{P3}/results/external_chapman_auc.csv", index=False)
pd.set_option("display.width", 250)
print(f"\n[5/6] External test on Chapman-Shaoxing (no retraining; {N_BOOT} bootstrap resamples for the 95% CI). "
      "Internal = the same model and class on the PTB-XL test fold (one-vs-rest; for NORM the external definition is "
      "stricter, see the top of the cell):")
print(ext.to_string(index=False))
sup = ext[ext.task.isin(SUPER)]
print("\n   Mean over the 5 superclass tasks: " + ", ".join(
    f"{m}: external {g['AUC external'].mean():.3f} vs internal {g['AUC internal (PTB-XL test)'].mean():.3f}"
    for m, g in sup.groupby("model", sort=False)))

# ---- 6) Figure and save ----
fig, ax = plt.subplots(figsize=(12, 4.2))
models = list(P) + ["lead II, 23 subclasses"]
tasks = list(TASKS) + list(NAMED)
col = {"12-lead": "#1f3864", "lead II, clean-trained": "#fda4af", "lead II, noise-trained": "#e11d48",
       "lead II, 23 subclasses": "#f59e0b"}
w = 0.2
for i, t in enumerate(tasks):
    g = ext[ext.task == t].reset_index(drop=True)
    for k, (_, r) in enumerate(g.iterrows()):
        x0 = i + (k - (len(g) - 1) / 2) * w
        ax.bar(x0, r["AUC external"], w, color=col[r["model"]], label=r["model"] if i in (0, len(TASKS)) and
               r["model"] not in ax.get_legend_handles_labels()[1] else None)
        ax.scatter(x0, r["AUC internal (PTB-XL test)"], marker="_", s=120, color="#111", zorder=3,
                   label="internal (PTB-XL test)" if (i, k) == (0, 0) else None)
ax.set_xticks(range(len(tasks)), tasks, fontsize=8.5)
ax.set_ylim(min(0.5, ext[['AUC external', 'AUC internal (PTB-XL test)']].min().min() - 0.03), 1.0)
ax.set_ylabel("ROC AUC")
ax.set_title("External test on Chapman-Shaoxing (bars) vs the PTB-XL test fold (black marks), no retraining",
             loc="left", fontsize=11)
ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{P3}/results/external_chapman_auc.png", dpi=150)
plt.show()
np.savez_compressed(f"{P3}/results/external_chapman_probs.npz", record=cm["record"].to_numpy().astype(str), est_snr=est,
                    p12=P["12-lead"], pII=P["lead II, clean-trained"], pII_noise=P["lead II, noise-trained"], sub23=P23)
print(f"[6/6] Saved: results/external_chapman_auc.csv, external_chapman_auc.png, external_chapman_probs.npz | total "
      f"{(time.time() - t_all) / 60:.0f} min")
