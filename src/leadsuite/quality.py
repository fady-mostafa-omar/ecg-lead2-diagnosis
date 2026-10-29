"""Signal quality of a 10-s single-lead strip, judged from the strip alone (no clean reference needed).
Seven fast features of the band-passed strip (shape of the amplitude distribution, where the power sits in the
spectrum, how periodic the strip is, its amplitude), and a gradient-boosting regressor trained to estimate the
strip's SNR in dB from them. A strip whose estimated SNR is below the gate threshold is not diagnosed
("signal too noisy"). A flat line (lead off) always fails the gate."""
import numpy as np
from scipy.stats import kurtosis, skew

FEATURES = ["kurtosis", "abs_skew", "power_0.5-3Hz", "power_5-15_over_5-40Hz", "power_15-40Hz", "periodicity",
            "log_std"]
CAP_DB = 30.0          # clean strips are labelled 30 dB when training the estimator
FLAT = -99.0           # estimated SNR returned for a flat line


def sqi_features(x, fs=100):
    """x: band-passed strips [n, L] (or [n, 1, L]) -> float32 [n, 7]. All but log_std ignore the amplitude."""
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    x = x - x.mean(1, keepdims=True)
    sd = x.std(1)
    n = x.shape[1]
    f = np.fft.rfftfreq(n, 1 / fs)
    P = np.abs(np.fft.rfft(x, axis=1)) ** 2

    def band(lo, hi):
        return P[:, (f >= lo) & (f < hi)].sum(1)

    tot = band(0.5, 40) + 1e-12
    nf = 2 ** int(np.ceil(np.log2(2 * n)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, nf, axis=1)) ** 2, nf, axis=1)[:, :n]
    ac = ac / (ac[:, :1] + 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.stack([kurtosis(x, axis=1), np.abs(skew(x, axis=1)), band(0.5, 3) / tot,
                        band(5, 15) / (band(5, 40) + 1e-12), band(15, 40) / tot,
                        ac[:, int(0.3 * fs):int(2.0 * fs)].max(1), np.log10(sd + 1e-9)], 1)
    out[sd < 1e-6] = np.nan
    return out.astype(np.float32)


def estimate_snr(gate, x, fs=100):
    """gate: {"model": regressor, "threshold": dB, ...}. Returns the estimated SNR (dB) of each strip."""
    F = sqi_features(x, fs)
    est = gate["model"].predict(F)
    est[np.isnan(F).all(1)] = FLAT
    return est


def accept(gate, x, fs=100):
    return estimate_snr(gate, x, fs) >= gate["threshold"]
