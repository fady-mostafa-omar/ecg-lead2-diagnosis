"""PTB-XL access: the compact npz written in Step 3.1, the label table and the official patient-wise split."""
import numpy as np
import pandas as pd

SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
NAMES = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "STTC": "ST/T change",
         "CD": "Conduction disturbance", "HYP": "Hypertrophy"}
FS = 100


def load_ptbxl(npz_path, meta_path):
    """Return (X int16 [n, 1000, 12], gain in units per mV, lead names, meta DataFrame indexed by ecg_id)."""
    d = np.load(npz_path)
    X, ids = d["X"], d["ecg_id"]
    meta = pd.read_csv(meta_path, index_col=0)
    assert (meta.index.to_numpy() == ids).all(), "npz and meta are not in the same order"
    return X, float(d["gain"]), [str(s) for s in d["leads"]], meta


def task_mask(meta):
    """ECGs with at least one diagnostic superclass (the PTB-XL superdiagnostic task)."""
    return (meta[SUPER].sum(1) > 0).to_numpy()


def label_matrix(meta):
    return meta[SUPER].to_numpy(dtype=int)
