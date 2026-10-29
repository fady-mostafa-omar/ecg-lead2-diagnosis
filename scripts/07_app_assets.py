# STEP 3.7a — The app (Gradio, CPU). Before it starts, two things it needs are computed here:
#   1) decision thresholds for the 5 groups and the 23 named findings, chosen on the VALIDATION fold (fold 9) by
#      maximising sensitivity + specificity, then scored once on the test fold (results/thresholds_test_performance.csv);
#   2) example ECGs: random PTB-XL test-fold ECGs (seed 0) with a single cardiologist label each, three copies with real
#      test-part noise added (12, 0 and -6 dB) (to show the quality gate), and a 60-s montage to show strip-by-strip navigation.
# CPU is enough (about 5 min). The app opens through a gradio.live link, or through Colab if that link fails.
# ---- 1) Setup: write the app ----
import os, sys, json, time, subprocess, importlib
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_curve

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
P3 = ROOT
NST = f"{P3}/data/nstdb"
SRC = f"{P3}/src"
os.makedirs(f"{P3}/app/examples", exist_ok=True)

sys.path_importer_cache.pop(SRC, None); importlib.invalidate_caches()
if SRC not in sys.path:
    sys.path.insert(0, SRC)
for m in [k for k in sys.modules if k.startswith("leadsuite")]:
    del sys.modules[m]
from leadsuite.data import SUPER, label_matrix, load_ptbxl, task_mask
from leadsuite.cnn import load_members, predict, preprocess
from leadsuite.noise import TYPES, bounds, load_nstdb

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))
MIN_VAL_POS = 20
print(f"[1/3] device {DEV}")

# ---- 2) Thresholds on the validation fold; performance at those thresholds on the test fold ----
t0 = time.time()
X, gain, leads, meta = load_ptbxl(f"{P3}/data/ptbxl100.npz", f"{P3}/data/ptbxl_meta.csv")
five, ck = load_members(f"{P3}/models/cnn_leadII_noise.pt")
sub, ck23 = load_members(f"{P3}/models/cnn_leadII_sub23.pt")
iII = leads.index("II")
scale = np.float32(ck["scale"][0])
Xf = preprocess(X[:, :, [iII]], gain) / scale
task, Y = task_mask(meta), label_matrix(meta)
split = meta["split"].to_numpy()
va, te = (np.where(task & (split == s))[0] for s in ("val", "test"))
rs = np.load(f"{P3}/results/pairs_sub23_probs_test.npz")
names23 = [str(s) for s in rs["sub23_names"]]
subs = meta["subclass"].fillna("").apply(lambda s: set(x for x in s.split(";") if x))
Y23 = np.array([[int(s in z) for s in names23] for z in subs])


def ens(members, x):
    return np.mean([predict(m.to(DEV), x, DEV) for m in members], 0)


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[int(np.argmax(tpr - fpr))], 0, 1))


pv5, pv23 = ens(five, Xf[va]), ens(sub, Xf[va])
thr5 = {c: youden(Y[va, j], pv5[:, j]) for j, c in enumerate(SUPER)}
thr23 = {s: youden(Y23[va, j], pv23[:, j]) for j, s in enumerate(names23) if Y23[va, j].sum() >= MIN_VAL_POS}
json.dump({"super": thr5, "sub23": thr23, "sub23_names": names23,
           "rule": "maximum of sensitivity + specificity on the validation fold (fold 9); "
                   f"named findings only when the validation fold has >= {MIN_VAL_POS} positives",
           "val_positives": {s: int(Y23[va, j].sum()) for j, s in enumerate(names23)}},
          open(f"{P3}/models/thresholds.json", "w"), indent=1)
rn = np.load(f"{P3}/results/noise_probs_test.npz")
assert (rn["ecg_id"] == meta.index.to_numpy()[te]).all() and (rs["ecg_id"] == meta.index.to_numpy()[te]).all()
pt5, pt23 = rn["II|noise-trained|none|None"], rs["sub23"]
rows = []
for kind, names, P, T, Yt in (("group", SUPER, pt5, thr5, Y[te]), ("named", names23, pt23, thr23, Y23[te])):
    for j, s in enumerate(names):
        if s not in T:
            continue
        y, f = Yt[:, j] == 1, P[:, j] >= T[s]
        rows.append({"kind": kind, "name": s, "threshold %": round(100 * T[s], 1), "test positives": int(y.sum()),
                     "sensitivity %": round(100 * (f & y).sum() / max(y.sum(), 1), 1),
                     "specificity %": round(100 * (~f & ~y).sum() / max((~y).sum(), 1), 1),
                     "PPV %": round(100 * (f & y).sum() / max(f.sum(), 1), 1)})
perf = pd.DataFrame(rows)
perf.to_csv(f"{P3}/results/thresholds_test_performance.csv", index=False)
pd.set_option("display.width", 200)
print(f"[2/3] Thresholds chosen on {len(va):,} validation ECGs ({time.time() - t0:.0f} s); scored on the {len(te):,} "
      "test ECGs (lead II, noise-trained ensemble and 23-subclass ensemble):")
print(perf.to_string(index=False))

# ---- 3) Examples: random test-fold ECGs with a single subclass label (seed 0) + noise demos + a 60-s montage ----
TARGETS = [("NORM", "Normal ECG"), ("IMI", "Inferior myocardial infarction"), ("AMI", "Anterior myocardial infarction"),
           ("CRBBB", "Complete right bundle branch block"), ("CLBBB", "Complete left bundle branch block"),
           ("LAFB/LPFB", "Left fascicular block"), ("_AVB", "AV block"), ("LVH", "Left ventricular hypertrophy"),
           ("NST_", "Non-specific ST change")]
rng = np.random.default_rng(0)
ex, meta_ex = {}, []
pid_test = set(meta["patient_id"].to_numpy()[te])
for k, (s, name) in enumerate(TARGETS):
    cand = [i for i in te if subs.iloc[i] == {s}] or [i for i in te if s in subs.iloc[i]]
    i = int(rng.choice(cand))
    key = f"ex{k}"
    ex[key] = X[i, :, iII].astype(np.int16)                    # gain 1000 per mV -> values in microvolts
    meta_ex.append({"key": key, "label": f"{name} · ECG {meta.index[i]}", "ecg_id": int(meta.index[i]),
                    "subclass": sorted(subs.iloc[i]), "noise": False, "fs": 100, "candidates": len(cand)})
noise = load_nstdb(NST)
nrng = np.random.default_rng(5)


def mix_segment(L=1000):
    """Raw (not band-passed) mix of the three noise types from the TEST part, each at equal in-band power."""
    out_raw, out_band = np.zeros(L), np.zeros(L)
    for t in TYPES:
        lo, hi = bounds(len(noise[t]), "test", L)
        st, ch = int(nrng.integers(lo, hi + 1)), int(nrng.integers(0, 2))
        raw = noise[t][st:st + L, ch].astype(np.float64)
        band = preprocess(raw[None, :, None].astype(np.float32), 1.0)[0, 0]
        p = np.sqrt((band ** 2).mean()) + 1e-12
        out_raw += raw / p
        out_band += band / p
    return out_raw, out_band


def add_raw(x_uv, snr_db):
    x = x_uv / 1000.0
    xb = preprocess(x[None, :, None].astype(np.float32), 1.0)[0, 0]
    nr, nb = mix_segment()
    k = np.sqrt((xb ** 2).mean() / ((nb ** 2).mean() * 10 ** (snr_db / 10)))
    return np.clip(np.round((x + k * nr) * 1000), -32768, 32767).astype(np.int16)


for snr, key in ((12, "noisy12"), (0, "noisy0"), (-6, "noisy6")):
    ex[key] = add_raw(ex["ex0"], snr)
    meta_ex.append({"key": key, "label": f"Normal ECG + real noise at {snr} dB · ECG {meta_ex[0]['ecg_id']}",
                    "ecg_id": meta_ex[0]["ecg_id"], "subclass": ["NORM"], "noise": True, "fs": 100})
mont = ["ex0", "ex1", "noisy0", "ex3", "noisy6", "ex5"]
ex["montage"] = np.concatenate([ex[k] for k in mont])
meta_ex.append({"key": "montage", "label": "Montage (60 s): 6 different test ECGs joined, 2 with added noise",
                "ecg_id": 0, "subclass": [], "noise": True, "fs": 100, "parts": mont})
np.savez_compressed(f"{P3}/app/examples/examples.npz", meta=json.dumps(meta_ex), **ex)
print(f"[3/3] {len(meta_ex)} examples saved to app/examples/examples.npz: " + "; ".join(m["label"] for m in meta_ex))
