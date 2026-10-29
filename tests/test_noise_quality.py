import os

import joblib
import numpy as np

from leadsuite.cnn import preprocess
from leadsuite.noise import GAP, TRAIN_FRAC, add_noise, augmenter, bounds
from leadsuite.quality import FEATURES, FLAT, estimate_snr, sqi_features
from helpers import synthetic_ecg

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_add_noise_hits_the_requested_snr():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 1, 1000)).astype(np.float32)
    nz = rng.normal(size=(50, 1000)).astype(np.float32)
    for snr in (12, 0, -6):
        y = add_noise(x, nz, snr)
        ratio = (x[:, 0] ** 2).mean(1) / ((y - x)[:, 0] ** 2).mean(1)
        assert np.allclose(10 * np.log10(ratio), snr, atol=1e-3)


def test_train_and_test_noise_never_overlap():
    n, L = 180_000, 1000
    lo_tr, hi_tr = bounds(n, "train", L)
    lo_te, hi_te = bounds(n, "test", L)
    assert hi_tr + L <= int(n * TRAIN_FRAC) - GAP // 2 and lo_te >= int(n * TRAIN_FRAC) + GAP // 2 and hi_te + L <= n


def test_augmenter_noises_about_half():
    import torch
    torch.manual_seed(0)
    bank = {t: np.random.default_rng(1).normal(size=(10, 1000)).astype(np.float32) for t in ("bw", "em", "ma", "mix")}
    xb = torch.ones(400, 1, 1000)
    changed = (augmenter(bank, p=0.5)(xb) != xb).any(dim=2).float().mean().item()
    assert 0.4 < changed < 0.6


def test_quality_features_and_flat_line():
    x = np.stack([synthetic_ecg(), np.zeros(1000)])
    F = sqi_features(preprocess(x[:, :, None].astype(np.float32), 1.0)[:, 0])
    assert F.shape == (2, len(FEATURES)) and np.isfinite(F[0]).all() and np.isnan(F[1]).all()


def test_shipped_gate_ranks_noise():
    gate = joblib.load(os.path.join(ROOT, "models", "quality_gate.joblib"))
    clean = synthetic_ecg(seed=2)
    noisy = clean + np.random.default_rng(3).normal(0, 1.0, 1000)
    xb = preprocess(np.stack([clean, noisy, np.zeros(1000)])[:, :, None].astype(np.float32), 1.0)[:, 0]
    est = estimate_snr(gate, xb / 0.2)
    assert est[0] > est[1] and est[2] == FLAT
