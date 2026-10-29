"""R-peak detection, reused from Project 1 (ecg-arrhythmia-detector): XQRS, then each detection is
moved to the largest |amplitude| within +-100 ms, so it works for either signal polarity."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from wfdb import processing


def bandpass(x, fs, lo=0.5, hi=40.0, order=3):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def detect_r_peaks(sig, fs):
    filt = bandpass(sig, fs)
    try:
        xqrs = processing.XQRS(sig=filt, fs=fs)
        xqrs.detect(verbose=False)
        det = np.asarray(xqrs.qrs_inds, dtype=int)
    except Exception:                       # very short or flat recordings
        det = np.array([], dtype=int)
    r = int(0.10 * fs)
    peaks = [max(d - r, 0) + int(np.argmax(np.abs(filt[max(d - r, 0):d + r + 1]))) for d in det]
    return np.unique(np.array(peaks, dtype=int))


def detect_one(args):
    """(signal, fs) -> peaks; a picklable wrapper for multiprocessing."""
    return detect_r_peaks(*args)
