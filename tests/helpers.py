"""Synthetic single-lead ECG for the tests (no real data needed)."""
import numpy as np


def synthetic_ecg(seconds=10, fs=100, rr=0.8, amp=1.0, seed=0):
    """Gaussian P, QRS and T waves plus a little noise, in mV."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    sig = rng.normal(0, 0.01, len(t))
    b = 0.4
    while b < seconds - 0.3:
        sig += amp * (0.12 * np.exp(-0.5 * ((t - b + 0.18) / 0.025) ** 2) + np.exp(-0.5 * ((t - b) / 0.012) ** 2)
                      - 0.15 * np.exp(-0.5 * ((t - b - 0.03) / 0.01) ** 2) + 0.3 * np.exp(-0.5 * ((t - b - 0.27) / 0.045) ** 2))
        b += rr
    return sig
