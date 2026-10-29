import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "app"))
import app as A
from helpers import synthetic_ecg


def test_upload_helpers(tmp_path):
    sig = synthetic_ecg(seconds=20, fs=500)
    f = tmp_path / "ecg.csv"
    np.savetxt(f, np.c_[0.3 * sig, sig], delimiter=",", header="I,II", comments="")
    names, header = A.read_columns(str(f))
    assert header and names == ["I", "II"] and A.default_column(names) == "II"
    assert np.allclose(A.read_signal(str(f), "II"), sig)
    assert len(A.to_100hz(sig, 500)) == 2000


def test_amplitude_check_never_inverts():
    sig = synthetic_ecg(seconds=10)
    fixed, notes = A.check_amplitude(-sig)                     # upside down but in mV: left as it is
    assert not notes and np.allclose(fixed, -sig)
    fixed, notes = A.check_amplitude(400 * sig + 2048)         # ADC counts: rescaled, with a warning
    assert len(notes) == 1 and np.percentile(np.abs(A.bandpass(fixed, 100)), 99.5) < 3


def test_models_examples_and_analysis():
    M = A.Models(os.path.join(ROOT, "models"))
    rel = A.reliability_table(os.path.join(ROOT, "results"))
    choices, ex = A.load_examples(os.path.join(ROOT, "app", "examples", "examples.npz"))
    assert len(choices) >= 10
    res = A.analyze(np.concatenate([synthetic_ecg(seconds=20), np.zeros(1000)]), M, rel)
    assert res["n"] == 3 and res["p5"].shape == (3, 5) and res["p23"].shape[0] == 3
    assert not res["ok"][2]                                     # the flat strip is refused
    assert "too noisy" in A.findings_html(res, 2)
    assert "Limitations" in A.build_about(os.path.join(ROOT, "results"))
