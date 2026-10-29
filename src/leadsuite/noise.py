"""Real ambulatory noise from the MIT-BIH Noise Stress Test Database (NSTDB): baseline wander (bw), electrode motion
(em) and muscle artefact (ma), recorded from a physically active volunteer with the electrodes placed where no ECG is
visible. Each noise record is split in time: the first 60% feeds training and validation, the last 40% is kept for the
test (with a 10-s gap), so no noise sample is shared between them.
SNR: both powers are measured on the band-passed (0.5-40 Hz) 10-s strip, i.e. on what the network sees. This is not
the QRS-based definition of PhysioNet's nst program, so these SNR values are not directly comparable to the ready-made
NSTDB records."""
import os

import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly

from .cnn import preprocess

TYPES = ["bw", "em", "ma"]
ALL = TYPES + ["mix"]
LABELS = {"bw": "baseline wander", "em": "electrode motion", "ma": "muscle artefact", "mix": "all three mixed"}
TRAIN_FRAC, GAP = 0.6, 1000


def load_nstdb(nst_dir, fs=100):
    """{type: float32 [samples, 2] in mV, resampled to fs}"""
    out = {}
    for t in TYPES:
        r = wfdb.rdrecord(os.path.join(nst_dir, t))
        g = np.gcd(int(fs), int(r.fs))
        out[t] = resample_poly(r.p_signal.astype(np.float64), int(fs) // g, int(r.fs) // g, axis=0).astype(np.float32)
    return out


def bounds(n, part, L=1000):
    """Allowed segment starts for one part of a noise record of n samples."""
    cut = int(n * TRAIN_FRAC)
    return (0, cut - GAP // 2 - L) if part == "train" else (cut + GAP // 2, n - L)


def _segments(x, part, n, rng, L=1000):
    lo, hi = bounds(len(x), part, L)
    st, ch = rng.integers(lo, hi + 1, n), rng.integers(0, x.shape[1], n)
    return np.stack([x[s:s + L, c] for s, c in zip(st, ch)])


def _band(seg):
    return preprocess(seg[:, :, None], 1.0)[:, 0, :]      # the same 0.5-40 Hz band-pass as the ECG, per 10-s strip


def _unit(x):
    return x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-12)


def noise_bank(noise, part, n, seed, L=1000):
    """n band-passed 10-s segments (random channel and start) of each type, plus 'mix' = the three at equal power,
    summed: {type: float32 [n, L]}."""
    rng = np.random.default_rng(seed)
    bank = {t: _band(_segments(noise[t], part, n, rng, L)) for t in TYPES}
    bank["mix"] = sum(_unit(bank[t]) for t in TYPES).astype(np.float32)
    return bank


def add_noise(x, nz, snr_db):
    """x: band-passed ECG [n, 1, L]; nz: band-passed noise [n, L]; snr_db: one value or one per strip."""
    ps = (x[:, 0] ** 2).mean(1)
    pn = (nz ** 2).mean(1) + 1e-12
    k = np.sqrt(ps / (pn * 10 ** (np.asarray(snr_db, np.float64) / 10)))
    return (x + k[:, None, None] * nz[:, None, :]).astype(np.float32)


def augmenter(bank, p=0.5, snr=(0.0, 24.0)):
    """Training-time noise for single-lead batches [n, 1, L]: with probability p, each strip gets a random training
    segment of a random type (bw, em, ma or mix) at a random SNR, uniform in dB."""
    cache = {}

    def aug(xb):
        dev = xb.device
        if dev not in cache:
            cache[dev] = torch.as_tensor(np.stack([bank[t] for t in ALL]), device=dev)
        B, n = cache[dev], xb.shape[0]
        nz = B[torch.randint(0, B.shape[0], (n,), device=dev), torch.randint(0, B.shape[1], (n,), device=dev)]
        ps = xb[:, 0].pow(2).mean(1)
        pn = nz.pow(2).mean(1) + 1e-12
        s = snr[0] + (snr[1] - snr[0]) * torch.rand(n, device=dev)
        k = torch.sqrt(ps / (pn * 10 ** (s / 10))) * (torch.rand(n, device=dev) < p)
        return xb + k[:, None, None] * nz[:, None, :]
    return aug
