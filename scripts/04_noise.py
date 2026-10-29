# STEP 3.4 — Real ambulatory noise (NSTDB: baseline wander, electrode motion, muscle artefact, and all three mixed)
# added to the lead I and lead II test ECGs at controlled SNR; then a CNN for each lead trained with that kind of noise
# (5 seeds each). Training noise and test noise come from different parts of each noise recording. Both leads get the
# same noise segments, so they can be compared directly (which electrode placement holds up better under noise).
# GPU recommended (about 25-30 min). Each seed is saved, so a re-run resumes.
# ---- 1) Setup ----
import os, sys, json, time, importlib
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
P3 = ROOT
NST = f"{P3}/data/nstdb"
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
from leadsuite.noise import ALL, LABELS, TRAIN_FRAC, add_noise, augmenter, load_nstdb, noise_bank

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS, EPOCHS = [0, 1, 2, 3, 4], 30
SNRS = [24, 18, 12, 6, 0, -6]
TRAIN_SNR, P_NOISY = (0.0, 24.0), 0.5
LEADS = {"I": "leadI", "II": "leadII"}
print(f"[1/6] Device: {DEV}" + ("" if DEV == "cuda" else "  ⚠️ no GPU: training will take many hours"))

# ---- 2) Lead I and lead II, the clean-trained ensembles of Step 3.3, and a check that they reproduce Step 3.3 ----
X, gain, leads, meta = load_ptbxl(f"{P3}/data/ptbxl100.npz", f"{P3}/data/ptbxl_meta.csv")
task, Y = task_mask(meta), label_matrix(meta)
split = meta["split"].to_numpy()
tr, va, te = (np.where(task & (split == s))[0] for s in ("train", "val", "test"))
ref = np.load(f"{P3}/results/cnn_probs_test.npz")
assert (ref["ecg_id"] == meta.index.to_numpy()[te]).all()


def ensemble(members, x):
    return np.mean([predict(m, x, DEV) for m in members], 0)


Xf, clean, scale, a0 = {}, {}, {}, {}
print(f"[2/6] {len(tr):,} train | {len(va):,} val | {len(te):,} test ECGs")
for L, tag in LEADS.items():
    mem, ck = load_members(f"{P3}/models/cnn_{tag}.pt")
    assert ck["leads"] == [leads.index(L)]
    scale[L] = np.float32(ck["scale"][0])
    Xf[L] = preprocess(X[:, :, ck["leads"]], gain) / scale[L]
    for m in mem:
        m.to(DEV)
    clean[L] = mem
    p = ensemble(mem, Xf[L][te])
    a0[L] = auc_per_class(Y[te], p)["macro"]
    same = np.abs(p - ref["p" + L]).max()
    print(f"   lead {L}: clean-trained ensemble on clean test ECGs, macro AUC {a0[L]:.3f} (Step 3.3: "
          f"{auc_per_class(Y[te], ref['p' + L])['macro']:.3f}; largest probability difference {same:.4f}) "
          f"{'✅' if same < 0.01 else '❌'}")
del X

# ---- 3) Noise: split each recording in time (training part / test part), fixed segments for the test ----
noise = load_nstdb(NST)
n_min = {t: len(v) / 100 / 60 for t, v in noise.items()}
train_bank = noise_bank(noise, "train", 4000, seed=1)              # drawn at random while training
val_bank = noise_bank(noise, "train", len(va), seed=2)             # validation: also from the training part
test_bank = noise_bank(noise, "test", len(te), seed=3)             # one fixed segment per test ECG and noise type
print(f"[3/6] Noise records: " + ", ".join(f"{t} {n_min[t]:.1f} min" for t in n_min) +
      f" | training part: first {TRAIN_FRAC:.0%} of each, test part: last {1 - TRAIN_FRAC:.0%} (10-s gap between)")
json.dump({"source": "MIT-BIH Noise Stress Test Database (bw, em, ma), both channels, resampled 360 -> 100 Hz",
           "train_part": f"first {TRAIN_FRAC:.0%} of each record", "test_part": f"last {1 - TRAIN_FRAC:.0%}",
           "gap_s": 10, "snr": "10*log10(P_ecg / P_noise), both on the 0.5-40 Hz band-passed 10-s strip",
           "mix": "bw + em + ma at equal power", "test_snr_db": SNRS, "train_snr_db": list(TRAIN_SNR),
           "train_noisy_fraction": P_NOISY, "leads": list(LEADS),
           "same_noise_for_both_leads": True,
           "seeds": {"train_bank": 1, "val_bank": 2, "test_bank": 3, "val_snr": 4}},
          open(f"{P3}/results/noise_protocol.json", "w"), indent=1)

# ---- 4) Train a noise-trained CNN per lead: half of the training strips get noise at 0-24 dB ----
rng = np.random.default_rng(4)
vt, vs = rng.integers(0, len(ALL), len(va)), rng.uniform(*TRAIN_SNR, len(va))
v_noise = np.stack([val_bank[ALL[t]][i] for i, t in enumerate(vt)])
aug = augmenter(train_bank, p=P_NOISY, snr=TRAIN_SNR)
noisy, spent, left = {}, [], len(LEADS) * len(SEEDS)
for L, tag in LEADS.items():
    VAL = (np.concatenate([Xf[L][va], add_noise(Xf[L][va], v_noise, vs)]), np.concatenate([Y[va], Y[va]]))
    noisy[L] = []
    for s in SEEDS:
        path = f"{P3}/models/cnn_seeds/{tag}_noise_seed{s}.pt"
        if os.path.exists(path):
            m, c = load_members(path)
            noisy[L].append(m[0])
            left -= 1
            print(f"   lead {L} noise-trained seed {s}: loaded (best epoch {c['best_epoch']}, validation {c['best_val']:.3f})")
            continue
        t0 = time.time()
        print(f"   lead {L} noise-trained seed {s} ", end="", flush=True)
        model, best, ep = train(Xf[L], Y, tr, va, epochs=EPOCHS, seed=s, device=DEV, aug=aug, val=VAL,
                                log=lambda msg: print(".", end="", flush=True))
        save_members(path, [model], {"seed": s, "best_epoch": ep, "best_val": best, "leads": [leads.index(L)],
                                     "scale": [float(scale[L])]})
        noisy[L].append(model)
        spent.append(time.time() - t0)
        left -= 1
        print(f" best epoch {ep}, validation macro AUC (clean + noisy) {best:.3f} | {spent[-1] / 60:.1f} min"
              + (f" | about {np.mean(spent) * left / 60:.0f} min left" if left else ""))
    save_members(f"{P3}/models/cnn_{tag}_noise.pt", noisy[L],
                 {"seeds": SEEDS, "epochs": EPOCHS, "leads": [leads.index(L)], "scale": [float(scale[L])],
                  "noise": "NSTDB training part, half of the strips, SNR uniform 0-24 dB",
                  "selection": "best macro AUC on fold 9, clean + noisy copies"})
    for m in noisy[L]:
        m.to(DEV)
print("[4/6] Saved: models/cnn_leadI_noise.pt, models/cnn_leadII_noise.pt")

# ---- 5) Test: every noise type at every SNR, both leads, both training recipes, the same noise segments ----
conds = [("none", None)] + [(t, s) for t in ALL for s in SNRS]
P, rows = {}, []
for L in LEADS:
    for t, s in conds:
        x = Xf[L][te] if t == "none" else add_noise(Xf[L][te], test_bank[t], s)
        for name, mem in (("clean-trained", clean[L]), ("noise-trained", noisy[L])):
            P[(L, name, t, s)] = p = ensemble(mem, x)
            a = auc_per_class(Y[te], p)
            row = {"lead": L, "model": name, "noise": t, "snr_db": s}
            row.update({c: round(a[c], 4) for c in SUPER})
            row["macro"] = round(a["macro"], 4)
            rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(f"{P3}/results/noise_auc.csv", index=False)


def cell(L, name, t, s):
    q = res[(res.lead == L) & (res.model == name) & (res.noise == t)]
    return (q if s is None else q[q.snr_db == s])["macro"].iloc[0]


tab = []
for L in LEADS:
    for t in ALL:
        for name in ("clean-trained", "noise-trained"):
            r = {"lead": L, "noise": LABELS[t], "model": name, "no noise": cell(L, name, "none", None)}
            r.update({f"{s} dB": cell(L, name, t, s) for s in SNRS})
            tab.append(r)
tab = pd.DataFrame(tab)
tab.to_csv(f"{P3}/results/noise_macro_auc_table.csv", index=False)
print(f"\n[5/6] Test fold ({len(te):,} ECGs), macro AUC by noise type and SNR (lower SNR = more noise):")
print(tab.round(3).to_string(index=False))
mix = res[(res.noise == "none") | ((res.noise == "mix") & res.snr_db.isin([12, 6, 0]))]
mix = mix.sort_values(["lead", "model"], kind="stable")
mix = mix.assign(condition=[("no noise" if n == "none" else f"mix {int(s)} dB") for n, s in zip(mix.noise, mix.snr_db)])
print("\n   Per diagnosis, all three noise types mixed:")
print(mix[["lead", "model", "condition"] + SUPER + ["macro"]].round(3).to_string(index=False))

pt = meta["patient_id"].to_numpy()[te]
keys = [("none", None)] + [("mix", s) for s in SNRS]
preds = {f"{L}|{n}|{t}|{s}": P[(L, n, t, s)] for L in LEADS for n in ("clean-trained", "noise-trained") for t, s in keys}
draws = patient_bootstrap(Y[te], preds, pt, n=1000, seed=0)
nb = len(draws[next(iter(draws))])
cols = SUPER + ["macro"]


def paired(a_key, b_key, label):
    d = draws[a_key] - draws[b_key]
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    r = dict(label)
    r.update({c: f"{d[:, j].mean():+.3f} ({lo[j]:+.3f} to {hi[j]:+.3f})" for j, c in enumerate(cols)})
    return r


cnd = lambda t, s: "no noise" if t == "none" else f"mix {s} dB"
diff = pd.DataFrame([paired(f"{L}|noise-trained|{t}|{s}", f"{L}|clean-trained|{t}|{s}", {"lead": L, "condition": cnd(t, s)})
                     for L in LEADS for t, s in keys])
diff.to_csv(f"{P3}/results/noise_auc_diff.csv", index=False)
print(f"\n   Noise-trained minus clean-trained, same lead ({nb} patient-level bootstrap resamples; an interval that "
      "excludes 0 is a real difference):")
print(diff.to_string(index=False))
ldiff = pd.DataFrame([paired(f"II|noise-trained|{t}|{s}", f"I|noise-trained|{t}|{s}", {"condition": cnd(t, s)})
                      for t, s in keys])
ldiff.to_csv(f"{P3}/results/noise_lead_diff.csv", index=False)
print("\n   Lead II minus lead I, both noise-trained, same noise segments:")
print(ldiff.to_string(index=False))
arrs = {f"{L}|{n}|{t}|{s}": v for (L, n, t, s), v in P.items()}
arrs["ecg_id"] = meta.index.to_numpy()[te]
np.savez_compressed(f"{P3}/results/noise_probs_test.npz", **arrs)

# ---- 6) Figures: AUC against SNR, and what the noise looks like ----
fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
color = {"I": "#2563eb", "II": "#e11d48"}
for ax, t in zip(axes, ALL):
    for L in LEADS:
        for name, ls, mk in (("clean-trained", "--", "o"), ("noise-trained", "-", "s")):
            ax.plot(SNRS, [cell(L, name, t, s) for s in SNRS], ls=ls, marker=mk, ms=3.5, lw=1.5, color=color[L],
                    alpha=0.6 if name == "clean-trained" else 1, label=f"lead {L}, {name}")
    ax.set_xlim(SNRS[0] + 1.5, SNRS[-1] - 1.5)
    ax.set_title(LABELS[t], fontsize=10, loc="left")
    ax.set_xlabel("SNR (dB)  →  more noise")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("macro ROC AUC (test fold)")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
fig.suptitle("Lead I and lead II CNNs under real recorded noise (MIT-BIH Noise Stress Test Database)", x=0.01,
             ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f"{P3}/results/noise_auc_vs_snr.png", dpi=150)
plt.show()

k = int(np.where((Y[te] == [1, 0, 0, 0, 0]).all(1))[0][0])
fig, axes = plt.subplots(4, 2, figsize=(15, 7), sharex=True)
tt = np.arange(Xf["II"].shape[2]) / 100
for col, L in enumerate(LEADS):
    for row, s in enumerate([None, 18, 6, 0]):
        ax = axes[row, col]
        x0 = Xf[L][te[k]:te[k] + 1]
        x = x0 if s is None else add_noise(x0, test_bank["mix"][k:k + 1], s)
        ax.plot(tt, x[0, 0] * scale[L], color="#111" if s is None else "#7c3aed", lw=0.8)
        ax.set_title(f"lead {L}: " + ("clean (band-passed 0.5-40 Hz)" if s is None else f"all three noise types, {s} dB"),
                     loc="left", fontsize=9.5)
        ax.set_ylabel("mV")
        ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("time (s)")
fig.suptitle(f"One normal test ECG (ECG {meta.index[te[k]]}), same noise segment on both leads", x=0.01, ha="left",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{P3}/results/noise_examples.png", dpi=120)
plt.show()
print("[6/6] Saved: results/noise_*.csv, noise_protocol.json, noise_probs_test.npz, noise_auc_vs_snr.png, "
      "noise_examples.png; models/cnn_leadI_noise.pt, models/cnn_leadII_noise.pt")
