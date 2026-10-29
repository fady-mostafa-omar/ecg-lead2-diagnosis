# STEP 3.1 — Data: PTB-XL (12-lead, 10 s, 100 Hz) + MIT-BIH Noise Stress Test Database (real ambulatory noise);
# labels, the official patient-wise split, and what lead I and lead II look like for each diagnosis
# ---- 1) Setup ----
import os, ast, glob, json, time, subprocess
import numpy as np, pandas as pd, wfdb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
P3 = ROOT
LOCAL = f"{P3}/data/raw/ptbxl"              # 21,799 small files
NST = f"{P3}/data/nstdb"
for d in (f"{P3}/data", f"{P3}/results", LOCAL, NST):
    os.makedirs(d, exist_ok=True)
S3 = "s3://physionet-open"
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
NAMES = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "STTC": "ST/T change",
         "CD": "Conduction disturbance", "HYP": "Hypertrophy"}
MIN_RECORDS = 21000


def s3_sync(src, dst, *include):
    inc = " ".join(f'--include "{p}"' for p in include)
    cmd = f'aws s3 sync --no-sign-request --only-show-errors {S3}/{src} {dst} ' + (f'--exclude "*" {inc}' if inc else "")
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


# ---- 2) Download PTB-XL v1.0.3: the 100 Hz records and the label tables ----
t0 = time.time()
r = s3_sync("ptb-xl/1.0.3/", LOCAL, "records100/*", "*.csv")
heads = glob.glob(f"{LOCAL}/**/records100/**/*.hea", recursive=True)
if len(heads) < MIN_RECORDS:
    print("AWS copy not complete (", r.stderr[-300:], ") -> downloading the zip from physionet.org (1.7 GB, slower)")
    url = ("https://physionet.org/static/published-projects/ptb-xl/"
           "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
    subprocess.run(f"wget -q -c -O {LOCAL}/ptbxl.zip {url} && unzip -q -o {LOCAL}/ptbxl.zip -d {LOCAL}/zip", shell=True)
    heads = glob.glob(f"{LOCAL}/**/records100/**/*.hea", recursive=True)
cands = [os.path.dirname(p) for p in glob.glob(f"{LOCAL}/**/ptbxl_database.csv", recursive=True)]
base = max(cands, key=lambda d: len(glob.glob(f"{d}/records100/*/*.hea")))
print(f"[1/5] PTB-XL: {len(heads):,} records (100 Hz) in {time.time() - t0:.0f} s")

# ---- 3) Labels: the 5 diagnostic superclasses (same aggregation as the PTB-XL benchmark) ----
db = pd.read_csv(f"{base}/ptbxl_database.csv", index_col="ecg_id")
db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
scp = pd.read_csv(f"{base}/scp_statements.csv", index_col=0)
diag = scp[scp["diagnostic"] == 1]
db["superclass"] = db["scp_codes"].apply(lambda d: sorted({diag.loc[k, "diagnostic_class"] for k in d if k in diag.index}))
db["subclass"] = db["scp_codes"].apply(lambda d: sorted({diag.loc[k, "diagnostic_subclass"] for k in d if k in diag.index}))
db["split"] = np.where(db["strat_fold"] <= 8, "train", np.where(db["strat_fold"] == 9, "val", "test"))
one_fold = db.groupby("patient_id")["strat_fold"].nunique().max() == 1
has = db["superclass"].apply(len) > 0
for c in SUPER:
    db[c] = db["superclass"].apply(lambda s, c=c: int(c in s))
print(f"[2/5] {len(db):,} ECGs from {db['patient_id'].nunique():,} patients | every patient in one fold only: "
      f"{'✅' if one_fold else '❌'} | ECGs with no diagnostic superclass (left out of the task): {int((~has).sum())}")
tab = db[has].groupby("split")[SUPER].sum().reindex(["train", "val", "test"])
tab["ECGs"] = db[has].groupby("split").size().reindex(["train", "val", "test"])
tab.loc["total"] = tab.sum()
print("   ECGs per superclass (one ECG can have several):")
print(tab.to_string())
combo = db[has]["superclass"].apply(lambda s: "+".join(s)).value_counts()
print(f"   ECGs with more than one superclass: {int((db[has]['superclass'].apply(len) > 1).sum()):,}; "
      f"most common combinations: " + ", ".join(f"{k} {v:,}" for k, v in combo.head(6).items()))
sub = pd.Series([s for lst in db[has]["subclass"] for s in lst]).value_counts()
print("   Most common subclasses: " + ", ".join(f"{k} {v:,}" for k, v in sub.head(12).items()))
tab.to_csv(f"{P3}/results/ptbxl_superclass_counts.csv")

# ---- 4) Compact copy on Drive: int16 samples of all ECGs (21,799 x 1,000 samples x 12 leads) ----
t0 = time.time()
X, gains, leads = np.zeros((len(db), 1000, 12), np.int16), set(), None
for i, (eid, fn) in enumerate(db["filename_lr"].items()):
    rec = wfdb.rdrecord(os.path.join(base, fn), physical=False)
    assert rec.fs == 100 and rec.d_signal.shape == (1000, 12), (eid, rec.fs, rec.d_signal.shape)
    leads = leads or rec.sig_name
    assert rec.sig_name == leads, (eid, rec.sig_name)
    X[i] = rec.d_signal
    gains |= set(rec.adc_gain)
    if (i + 1) % 5000 == 0:
        print(f"   read {i + 1:,}/{len(db):,} ({time.time() - t0:.0f} s)", flush=True)
assert len(gains) == 1, gains
np.savez_compressed(f"{P3}/data/ptbxl100.npz", X=X, ecg_id=db.index.to_numpy(), gain=float(gains.pop()),
                    leads=np.array(leads))
meta = db[["patient_id", "age", "sex", "strat_fold", "split"] + SUPER].copy()
meta["subclass"] = db["subclass"].apply(";".join)
meta["scp_codes"] = db["scp_codes"].apply(json.dumps)
meta.to_csv(f"{P3}/data/ptbxl_meta.csv")
print(f"[3/5] Saved data/ptbxl100.npz ({os.path.getsize(f'{P3}/data/ptbxl100.npz') / 1e6:.0f} MB) and data/ptbxl_meta.csv "
      f"in {time.time() - t0:.0f} s | leads: {', '.join(leads)}")

# ---- 5) What lead I and lead II show for each superclass (training folds only; one clear example each) ----
iI, iII = leads.index("I"), leads.index("II")
g = 1000.0
fig, axes = plt.subplots(5, 2, figsize=(14, 11), sharex=True)
t = np.arange(1000) / 100
for row, c in enumerate(SUPER):
    pick = db[(db["superclass"].apply(lambda s, c=c: s == [c])) & (db["split"] == "train")
              & (db["validated_by_human"] == True)].index[0]
    k = db.index.get_loc(pick)
    codes = ", ".join(db.loc[pick, "scp_codes"])
    for col, (li, ln) in enumerate(((iI, "lead I"), (iII, "lead II"))):
        ax = axes[row, col]
        ax.plot(t, X[k, :, li] / g, color="#111", lw=0.9)
        ax.set_title(f"{NAMES[c]}: {ln}  (ECG {pick}; {codes})", loc="left", fontsize=9.5)
        ax.set_ylabel("mV")
        ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(f"{P3}/results/examples_lead1_lead2.png", dpi=120)
plt.show()
print("[4/5] Figure saved: results/examples_lead1_lead2.png")

# ---- 6) MIT-BIH Noise Stress Test Database: real noise recorded with the same electrodes as an ECG ----
s3_sync("nstdb/1.0.0/", NST)
if not glob.glob(f"{NST}/*.hea"):                      # not on the AWS mirror: PhysioNet directly
    wfdb.dl_database("nstdb", NST)
fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
for ax, (rec, what) in zip(axes, (("bw", "baseline wander"), ("em", "electrode motion"), ("ma", "muscle artefact"))):
    r = wfdb.rdrecord(os.path.join(NST, rec))
    x = r.p_signal[: int(10 * r.fs), 0]
    ax.plot(np.arange(len(x)) / r.fs, x, color="#7c3aed", lw=0.8)
    ax.set_title(f"{what} ('{rec}'): {r.sig_len / r.fs / 60:.0f} min, {r.fs} Hz, {r.n_sig} channels", loc="left", fontsize=9.5)
    ax.set_ylabel("mV")
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(f"{P3}/results/nstdb_noise_examples.png", dpi=120)
plt.show()
print(f"[5/5] Noise Stress Test DB: {len(glob.glob(f'{NST}/*.hea'))} records in {NST}")
