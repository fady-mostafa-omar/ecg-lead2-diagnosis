import numpy as np
import torch

from leadsuite.cnn import LeadNet, load_members, predict, preprocess, save_members
from leadsuite.metrics import auc_per_class, macro_auc, patient_bootstrap


def test_preprocess_shape_and_band():
    rng = np.random.default_rng(0)
    X = (rng.normal(0, 100, (3, 1000, 12)) + 500).astype(np.int16)        # DC offset must be removed
    out = preprocess(X, 1000.0)
    assert out.shape == (3, 12, 1000) and out.dtype == np.float32
    assert np.abs(out.mean(axis=2)).max() < 0.05


def test_leadnet_shapes():
    for leads, classes in ((1, 5), (2, 5), (12, 5), (1, 23)):
        m = LeadNet(leads, classes).eval()
        with torch.no_grad():
            y = m(torch.zeros(4, leads, 1000))
        assert y.shape == (4, classes)


def test_save_load_roundtrip(tmp_path):
    m = LeadNet(1, 23).eval()
    p = tmp_path / "m.pt"
    save_members(str(p), [m], {"leads": [1], "scale": [0.2]})
    mm, ck = load_members(str(p))
    x = np.random.default_rng(0).normal(size=(2, 1, 1000)).astype(np.float32)
    assert ck["n_classes"] == 23 and np.allclose(predict(m, x, "cpu"), predict(mm[0], x, "cpu"), atol=1e-6)


def test_metrics():
    rng = np.random.default_rng(1)
    Y = rng.integers(0, 2, (300, 5))
    P = np.clip(Y * 0.6 + rng.random((300, 5)) * 0.5, 0, 1)
    assert abs(macro_auc(Y, P) - auc_per_class(Y, P)["macro"]) < 1e-12
    d = patient_bootstrap(Y, {"a": P}, np.repeat(np.arange(100), 3), n=20, seed=0)
    assert d["a"].shape[1] == 6 and 0.5 < d["a"][:, -1].mean() <= 1
