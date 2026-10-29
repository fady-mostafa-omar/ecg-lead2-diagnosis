"""Hand-crafted features of one 10-s ECG: beat timing, and the median beat of each lead (template samples plus
amplitudes and intervals that cardiologists read: R and S/Q size, QRS width, ST level, T wave, P wave)."""
import numpy as np

from .detect import bandpass, detect_r_peaks

PRE, POST = 0.25, 0.45                      # s before / after the R peak
RR_KEYS = ["n_beats", "hr_mean", "rr_sd", "rmssd", "rr_cv", "rr_min", "rr_max"]
MORPH_KEYS = ["r_amp", "neg_amp", "qrs_width", "st_level", "t_max", "t_min", "p_ptp", "beat_corr"]


def rr_features(peaks, fs):
    out = dict.fromkeys(RR_KEYS, np.nan)
    out["n_beats"] = len(peaks)
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.25) & (rr < 2.5)]
    if len(rr) >= 2:
        d = np.diff(rr)
        out.update(hr_mean=60 / rr.mean(), rr_sd=rr.std(), rmssd=np.sqrt(np.mean(d ** 2)), rr_cv=rr.std() / rr.mean(),
                   rr_min=rr.min(), rr_max=rr.max())
    return out


def lead_features(x, peaks, fs):
    """x: one band-passed lead (mV). Returns (median beat relative to the PR baseline, morphology dict)."""
    pre, post = int(round(PRE * fs)), int(round(POST * fs))
    keep = peaks[(peaks >= pre) & (peaks + post < len(x))]
    if len(keep) < 2:
        return np.full(pre + post, np.nan), dict.fromkeys(MORPH_KEYS, np.nan)
    W = x[keep[:, None] + np.arange(-pre, post)]
    tmpl = np.median(W, 0)
    s = lambda a, b: slice(pre + int(round(a * fs)), pre + int(round(b * fs)) + 1)   # window in s around R
    t = tmpl - np.median(tmpl[s(-0.09, -0.05)])                                     # PR segment = baseline
    qrs = t[s(-0.06, 0.06)]
    big = max(np.abs(qrs).max(), 1e-6)
    corr = [np.corrcoef(w, tmpl)[0, 1] for w in W] if np.std(tmpl) > 0 else [np.nan]
    out = {"r_amp": t[pre], "neg_amp": qrs.min(), "qrs_width": np.sum(np.abs(t[s(-0.07, 0.07)]) > 0.3 * big) / fs,
           "st_level": t[pre + int(round(0.08 * fs))], "t_max": t[s(0.15, 0.40)].max(), "t_min": t[s(0.15, 0.40)].min(),
           "p_ptp": np.ptp(t[s(-0.25, -0.10)]), "beat_corr": float(np.nanmean(corr))}
    return t, out


def feature_names(leads, fs=100):
    n = int(round(PRE * fs)) + int(round(POST * fs))
    names = list(RR_KEYS)
    for ld in leads:
        names += [f"{ld}_t{k}" for k in range(n)] + [f"{ld}_{k}" for k in MORPH_KEYS]
    return names


def ecg_features(sig, fs, detect_lead, use_leads):
    """sig: [samples, leads] in mV. R peaks are found on `detect_lead` and used for every lead in `use_leads`."""
    filt = np.stack([bandpass(sig[:, i], fs) for i in range(sig.shape[1])], 1)
    peaks = detect_r_peaks(sig[:, detect_lead], fs)
    vec = list(rr_features(peaks, fs).values())
    for i in use_leads:
        t, m = lead_features(filt[:, i], peaks, fs)
        vec += list(t) + list(m.values())
    return np.array(vec, dtype=np.float32)


def features_one(args):
    """(signal, fs, detect_lead, use_leads) -> feature vector; a picklable wrapper for multiprocessing."""
    return ecg_features(*args)
