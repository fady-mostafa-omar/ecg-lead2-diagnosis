# STEP 3.3 — Deep model: a residual 1D-CNN on the raw 10-s ECG for 12 leads, lead II and lead I; 5 seeds each, the
# 5-seed ensemble fixed in advance; epoch chosen on fold 9; scored once on fold 10 against the Step 3.2 baseline.
# GPU needed: Runtime -> Change runtime type -> T4 GPU (about 30-40 min). Each seed is saved, so a re-run resumes.
# ---- 1) Setup ----
import os, sys, time, importlib
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

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
from leadsuite.metrics import auc_per_class, patient_bootstrap
from leadsuite.cnn import load_members, predict, preprocess, save_members, train

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS, EPOCHS = [0, 1, 2, 3, 4], 30
print(f"[1/5] Device: {DEV}" + ("" if DEV == "cuda" else "  ⚠️ no GPU: this step will take many hours"))

# ---- 2) Data: band-pass 0.5-40 Hz, keep mV, one global scale per lead (training ECGs only) ----
X, gain, leads, meta = load_ptbxl(f"{P3}/data/ptbxl100.npz", f"{P3}/data/ptbxl_meta.csv")
Xf = preprocess(X, gain)
del X
task, Y = task_mask(meta), label_matrix(meta)
split = meta["split"].to_numpy()
tr, va, te = (np.where(task & (split == s))[0] for s in ("train", "val", "test"))
scale = Xf[tr].std(axis=(0, 2)).astype(np.float32)
Xf /= scale[None, :, None]
CONFIG = {"12-lead": list(range(12)), "lead II": [leads.index("II")], "lead I": [leads.index("I")]}
print(f"[2/5] {Xf.shape[0]:,} ECGs x {Xf.shape[1]} leads x {Xf.shape[2]} samples | train {len(tr):,} | val {len(va):,} | "
      f"test {len(te):,}")

# ---- 3) Train 5 seeds per configuration (saved one by one) ----
members, spent, left = {}, [], len(CONFIG) * len(SEEDS)
for cfg, li in CONFIG.items():
    Xc = np.ascontiguousarray(Xf[:, li, :])
    members[cfg] = []
    tag = cfg.replace(" ", "").replace("-", "")
    for s in SEEDS:
        path = f"{P3}/models/cnn_seeds/{tag}_seed{s}.pt"
        if os.path.exists(path):
            m, ck = load_members(path)
            members[cfg].append(m[0])
            left -= 1
            print(f"   {cfg} seed {s}: loaded (best epoch {ck['best_epoch']}, validation {ck['best_val']:.3f})")
            continue
        t0 = time.time()
        print(f"   {cfg} seed {s} ", end="", flush=True)
        model, best, ep = train(Xc, Y, tr, va, epochs=EPOCHS, seed=s, device=DEV,
                                log=lambda msg: print(".", end="", flush=True))
        save_members(path, [model], {"seed": s, "best_epoch": ep, "best_val": best, "leads": li,
                                     "scale": scale[li].tolist()})
        members[cfg].append(model)
        spent.append(time.time() - t0)
        left -= 1
        print(f" best epoch {ep}, validation macro AUC {best:.3f} | {spent[-1] / 60:.1f} min"
              + (f" | about {np.mean(spent) * left / 60:.0f} min left" if left else ""))
    save_members(f"{P3}/models/cnn_{tag}.pt", members[cfg], {"seeds": SEEDS, "epochs": EPOCHS, "leads": li,
                                                             "scale": scale[li].tolist(),
                                                             "selection": "best validation macro AUC (fold 9)"})
n_par = {cfg: sum(p.numel() for p in members[cfg][0].parameters()) for cfg in CONFIG}
print(f"[3/5] Parameters per network: " + ", ".join(f"{k} {v:,}" for k, v in n_par.items()))

# ---- 4) Test fold: each seed alone, the 5-seed ensemble, and the baseline of Step 3.2 ----
probs, seed_rows = {}, []
for cfg, li in CONFIG.items():
    Xc = np.ascontiguousarray(Xf[:, li, :])
    ps = []
    for s, m in zip(SEEDS, members[cfg]):
        p = predict(m.to(DEV), Xc[te], DEV)
        m.cpu()
        ps.append(p)
        seed_rows.append({"config": cfg, "seed": s, "test macro AUC": auc_per_class(Y[te], p)["macro"]})
    probs[cfg] = np.mean(ps, 0)
seeds = pd.DataFrame(seed_rows)
b = np.load(f"{P3}/results/baseline_probs.npz")
assert (b["ecg_id"] == meta.index.to_numpy()).all()
base = {"12-lead": b["p12"][te], "lead II": b["pII"][te], "lead I": b["pI"][te]}
rows = []
for cfg in CONFIG:
    for name, P in ((f"CNN, single (mean ± SD, {len(SEEDS)} seeds)", None), (f"CNN, {len(SEEDS)}-seed ensemble", probs[cfg]),
                    ("Gradient boosting (Step 3.2)", base[cfg])):
        row = {"config": cfg, "model": name}
        if P is None:
            sv = seeds[seeds["config"] == cfg]["test macro AUC"]
            row["macro AUC"] = f"{sv.mean():.3f} ± {sv.std(ddof=1):.3f}"
        else:
            a = auc_per_class(Y[te], P)
            row.update({c: round(a[c], 3) for c in SUPER})
            row["macro AUC"] = f"{a['macro']:.3f}"
        rows.append(row)
auc = pd.DataFrame(rows)
print(f"\n[4/5] Test fold ({len(te):,} ECGs), ROC AUC:")
print(auc.astype(object).where(auc.notna(), "").to_string(index=False))

pt = meta["patient_id"].to_numpy()[te]
preds = {f"CNN {k}": v for k, v in probs.items()}
preds.update({f"GB {k}": v for k, v in base.items()})
draws = patient_bootstrap(Y[te], preds, pt, n=1000, seed=0)
cols = SUPER + ["macro"]
diff = []
for a_, b_ in (("CNN 12-lead", "GB 12-lead"), ("CNN lead II", "GB lead II"), ("CNN lead I", "GB lead I"),
               ("CNN lead II", "CNN 12-lead"), ("CNN lead II", "CNN lead I")):
    d = draws[a_] - draws[b_]
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    row = {"comparison": f"{a_} minus {b_}"}
    row.update({c: f"{d[:, j].mean():+.3f} ({lo[j]:+.3f} to {hi[j]:+.3f})" for j, c in enumerate(cols)})
    diff.append(row)
ci = []
for k in ("CNN 12-lead", "CNN lead II", "CNN lead I"):
    lo, hi = np.percentile(draws[k], [2.5, 97.5], axis=0)
    row = {"model": k}
    row.update({c: f"{lo[j]:.3f}-{hi[j]:.3f}" for j, c in enumerate(cols)})
    ci.append(row)
print(f"\n   95% confidence intervals of the ensembles ({len(draws['CNN 12-lead'])} patient-level bootstrap resamples):")
print(pd.DataFrame(ci).to_string(index=False))
print("\n   Paired differences in AUC (mean and 95% interval; an interval that excludes 0 is a real difference):")
print(pd.DataFrame(diff).to_string(index=False))
print("   Reported for context (Strodthoff et al., 12-lead, same test fold): macro AUC 0.921-0.930")

# ---- 5) Save, speed, and one figure ----
seeds.round(4).to_csv(f"{P3}/results/cnn_seeds.csv", index=False)
auc.to_csv(f"{P3}/results/cnn_auc.csv", index=False)
pd.DataFrame(ci).to_csv(f"{P3}/results/cnn_auc_ci.csv", index=False)
pd.DataFrame(diff).to_csv(f"{P3}/results/cnn_auc_diff.csv", index=False)
np.savez_compressed(f"{P3}/results/cnn_probs_test.npz", ecg_id=meta.index.to_numpy()[te], p12=probs["12-lead"],
                    pII=probs["lead II"], pI=probs["lead I"])
torch.set_num_threads(max(1, os.cpu_count() or 1))
xs = np.ascontiguousarray(Xf[te[:200], 1:2, :])
for m in members["lead II"]:
    predict(m, xs[:8], "cpu")
t0 = time.time()
for m in members["lead II"]:
    predict(m, xs, "cpu")
ms = 1000 * (time.time() - t0) / len(xs)
print(f"\n[5/5] Lead II ensemble on CPU: {ms:.1f} ms per 10-s ECG ({len(SEEDS)} networks)")
fig, ax = plt.subplots(figsize=(10, 4.2))
colors = {"12-lead": "#1f3864", "lead II": "#e11d48", "lead I": "#9ca3af"}
for j, cfg in enumerate(CONFIG):
    a = auc_per_class(Y[te], probs[cfg])
    lo, hi = np.percentile(draws[f"CNN {cfg}"], [2.5, 97.5], axis=0)
    gb = auc_per_class(Y[te], base[cfg])
    x = np.arange(len(cols)) + (j - 1) * 0.26
    y = [a[c] for c in SUPER] + [a["macro"]]
    ax.bar(x, y, 0.26, color=colors[cfg], label=f"CNN {cfg}")
    ax.errorbar(x, y, yerr=[np.array(y) - lo, hi - np.array(y)], fmt="none", ecolor="#111", lw=0.8, capsize=2)
    ax.scatter(x, [gb[c] for c in SUPER] + [gb["macro"]], marker="_", s=180, color="#f59e0b", zorder=3,
               label="gradient boosting (Step 3.2)" if j == 0 else None)
ax.set_xticks(range(len(cols)), [NAMES[c] for c in SUPER] + ["Macro AUC"], fontsize=8.5)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ROC AUC (test fold)")
ax.set_title("1D-CNN ensemble (bars) vs gradient boosting (orange marks), per lead configuration", fontsize=11, loc="left")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{P3}/results/cnn_auc_by_class.png", dpi=150)
plt.show()
print("Saved: results/cnn_*.csv, results/cnn_probs_test.npz, models/cnn_*.pt, results/cnn_auc_by_class.png")
