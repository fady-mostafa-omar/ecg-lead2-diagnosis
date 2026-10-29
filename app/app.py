"""ECG Lead II Diagnosis — Gradio app (CPU). Screens 10-s single-lead ECG strips (lead II) for the five PTB-XL
diagnostic groups, names the specific findings that one lead detects reliably, and refuses strips that are too
noisy to read. Research and education tool, not a medical device."""
import inspect
import json
import os
import sys
import tempfile
from fractions import Fraction

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "src"), os.path.join(HERE, "src")):
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
from leadsuite.cnn import load_members, predict, preprocess
from leadsuite.data import SUPER
from leadsuite.detect import bandpass, detect_r_peaks
from leadsuite.quality import estimate_snr

FS, STRIP, MAX_STRIPS = 100, 10, 60
PLAIN = {"NORM": "Normal ECG", "MI": "Myocardial infarction (heart attack) pattern",
         "STTC": "ST/T change (possible ischaemia)", "CD": "Conduction disturbance (bundle branch or AV block)",
         "HYP": "Hypertrophy (enlarged heart chamber)"}
SUBNAME = {"NORM": "Normal ECG", "IMI": "Inferior myocardial infarction", "AMI": "Anterior myocardial infarction",
           "LMI": "Lateral myocardial infarction", "PMI": "Posterior myocardial infarction", "STTC": "ST/T change",
           "NST_": "Non-specific ST change", "ISC_": "Non-specific ischaemia", "ISCA": "Ischaemia, anterior leads",
           "ISCI": "Ischaemia, inferior leads", "LAFB/LPFB": "Left anterior/posterior fascicular block",
           "IRBBB": "Incomplete right bundle branch block", "_AVB": "AV block (1st, 2nd or 3rd degree)",
           "IVCD": "Non-specific intraventricular conduction delay", "CRBBB": "Complete right bundle branch block",
           "CLBBB": "Complete left bundle branch block", "ILBBB": "Incomplete left bundle branch block",
           "WPW": "Wolff-Parkinson-White pattern", "LVH": "Left ventricular hypertrophy",
           "LAO/LAE": "Left atrial enlargement", "RAO/RAE": "Right atrial enlargement",
           "RVH": "Right ventricular hypertrophy", "SEHYP": "Septal hypertrophy"}
GRID_MAJOR, GRID_MINOR, PAPER = "rgba(220,60,60,0.55)", "rgba(220,60,60,0.18)", "#fff6f6"
FONT = dict(family="Inter, sans-serif", size=12)
C_OK, C_BAD, C_NOISY, C_WARN = "#2e9e5b", "#dc2626", "#9ca3af", "#d68910"

ICON = {
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "heart": '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "signal": '<path d="M2 12h3l3-8 4 16 3-8h7"/>',
    "usercheck": '<path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><polyline points="17 11 19 13 23 9"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
}

CSS = """
.title {text-align:center; font-size:2.1rem; font-weight:800; margin:8px 0 12px; letter-spacing:-0.02em;}
.sec {font-weight:700; font-size:1.05rem; margin:20px 0 8px; padding-bottom:6px;
      border-bottom:2px solid var(--border-color-primary);}
.kpi-row {display:flex; gap:12px; flex-wrap:wrap; margin:6px 0 12px;}
.kpi {flex:1 1 170px; display:flex; gap:12px; align-items:center; border:1px solid var(--border-color-primary);
      border-radius:14px; padding:12px 14px; background:var(--background-fill-secondary);}
.kpi .ic {flex:0 0 42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center;
          background:rgba(225,29,72,.08); color:#e11d48;}
.kpi.ok .ic {background:rgba(46,158,91,.12); color:#2e9e5b;}
.kpi.warn .ic {background:rgba(243,156,18,.15); color:#d68910;}
.kpi.bad .ic {background:rgba(220,38,38,.12); color:#dc2626;}
.kpi .ic svg {width:22px; height:22px; stroke:currentColor; fill:none; stroke-width:2;
              stroke-linecap:round; stroke-linejoin:round;}
.kpi .v {font-size:1.4rem; font-weight:700; line-height:1.15;}
.kpi .l {font-size:.8rem; opacity:.8;}
.kpi .s {font-size:.72rem; opacity:.6;}
.find {border:1px solid var(--border-color-primary); border-radius:14px; padding:10px 16px; margin:6px 0 12px;
       background:var(--background-fill-secondary);}
.find li {margin:4px 0;}
.badge {font-size:.72rem; padding:2px 8px; border-radius:999px; margin-left:6px; border:1px solid currentColor;}
.b-high {color:#2e9e5b;} .b-mod {color:#d68910;} .b-low {color:#9ca3af;}
.noisy {border:2px solid #9ca3af; border-radius:14px; padding:12px 16px; margin:6px 0 12px;}
.req {display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 12px;}
.req span {padding:4px 10px; border-radius:999px; font-size:.82rem;
           border:1px solid var(--border-color-primary); background:var(--background-fill-secondary);}
.step {font-weight:700; font-size:1rem; margin:4px 0 2px;}
.pipe {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:8px 0 16px;}
.chip {padding:6px 12px; border-radius:999px; border:1px solid var(--border-color-primary);
       background:var(--background-fill-secondary); font-size:.9rem;}
.arrow {opacity:.5;}
.note {font-size:.85rem; opacity:.75;}
"""
DESC = ("Reads 10-second strips of a single-lead ECG (lead II), checks first whether each strip is clean enough to "
        "read, and then screens it for five groups of ECG diagnoses: normal, myocardial infarction, ST/T change, "
        "conduction disturbance and hypertrophy. It also names specific findings, but only the ones a single lead "
        "detects reliably. It cannot replace a 12-lead ECG: it points to who needs one.")
REQ_HTML = ('<div class="req"><span>Single-lead ECG, lead II placement</span><span>CSV · millivolts</span>'
            '<span>Any sampling rate ≥ 100 Hz</span><span>10 s to 10 min (10-s strips)</span>'
            '<span>Research tool · not a medical device</span></div>')
PIPE_HTML = ('<div class="pipe">' + '<span class="arrow">→</span>'.join(
    f'<span class="chip">{s}</span>' for s in
    ["Raw ECG", "Resample to 100 Hz", "Band-pass 0.5-40 Hz", "10-s strips", "Quality gate (estimated SNR)",
     "5 × noise-trained 1D-CNN: 5 groups", "3 × 1D-CNN: 23 named findings", "Thresholds chosen on validation"]) + "</div>")
INPUT_MD = """
### Which recordings work?
| Source | Works? |
|---|---|
| 12-lead hospital ECG | ✅ Choose the lead II column (the model was trained on PTB-XL lead II) |
| Holter or chest-electrode recording, lead II-like placement | ✅ Expected to work; tested on lead II of another hospital (Chapman-Shaoxing) |
| AD8232 or another 3-electrode board | ⚠️ Place the electrodes for lead II; accuracy on these boards has **not** been measured yet |
| Lead I or other leads | ⚠️ Not what the model was trained on; results will be less reliable |

Polarity: **no automatic inversion.** The polarity of lead II carries diagnostic information (for example in left anterior
fascicular block the QRS in lead II is mostly negative), so an upside-down signal is analyzed as it is. If the R waves
point down because two electrodes are swapped, swap them back.

Amplitude: **it matters.** Hypertrophy is judged partly from QRS voltage, so the file must be in millivolts. If the values
do not look like millivolts (for example raw ADC counts), the app rescales them and says so; voltage-based findings
(hypertrophy) are then unreliable.
"""


def sec(title):
    return f'<div class="sec">{title}</div>'


def mmss(s):
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def cards_html(items):
    out = []
    for icon, label, value, sub, status in items:
        out.append(f'<div class="kpi {status}"><div class="ic"><svg viewBox="0 0 24 24">{ICON[icon]}</svg></div>'
                   f'<div><div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div></div>')
    return '<div class="kpi-row">' + "".join(out) + "</div>"


def read_columns(path):
    head = pd.read_csv(path, nrows=5, header=None)
    if pd.to_numeric(head.iloc[0], errors="coerce").isna().all():
        return [str(v) for v in head.iloc[0]], True
    return [f"Column {i + 1}" for i in range(head.shape[1])], False


def default_column(names):
    for n in names:
        if n.strip().upper().replace("_", " ") in ("II", "MLII", "LEAD II", "ECG II", "ECG"):
            return n
    return names[-1]


def read_signal(path, column):
    names, has_header = read_columns(path)
    df = pd.read_csv(path, header=0 if has_header else None)
    idx = names.index(column) if column in names else len(names) - 1
    return pd.to_numeric(df.iloc[:, idx], errors="coerce").dropna().to_numpy(float)


def to_100hz(sig, fs):
    if abs(fs - FS) < 1e-6:
        return np.asarray(sig, float)
    fr = Fraction(FS / fs).limit_denominator(1000)
    return resample_poly(np.asarray(sig, float), fr.numerator, fr.denominator)


def check_amplitude(sig):
    """Rescale signals that are clearly not in millivolts. No inversion (polarity is diagnostic in lead II)."""
    f = bandpass(sig, FS)
    amp = np.percentile(np.abs(f), 99.5)
    if amp > 8 or amp < 0.08:
        k = 1.2 / max(amp, 1e-9)
        return sig * k, [f"amplitude rescaled ×{k:.3g} because the values did not look like millivolts; "
                         "hypertrophy (voltage) findings are unreliable"]
    return sig, []


# ---------------------------------------------------------------- models
class Models:
    def __init__(self, models_dir):
        self.five, ck = load_members(os.path.join(models_dir, "cnn_leadII_noise.pt"))
        self.sub, ck23 = load_members(os.path.join(models_dir, "cnn_leadII_sub23.pt"))
        self.scale = np.float32(ck["scale"][0])
        assert abs(float(ck23["scale"][0]) - float(self.scale)) < 1e-6
        self.gate = joblib.load(os.path.join(models_dir, "quality_gate.joblib"))
        th = json.load(open(os.path.join(models_dir, "thresholds.json")))
        self.sub_names, self.thr5, self.thr23 = th["sub23_names"], th["super"], th["sub23"]

    def run(self, strips_mv):
        """strips_mv: [n, 1000] lead II in mV at 100 Hz."""
        xb = preprocess(strips_mv[:, :, None].astype(np.float32), 1.0) / self.scale      # [n, 1, 1000]
        est = estimate_snr(self.gate, xb[:, 0])
        with torch.no_grad():
            p5n = np.stack([predict(m, xb, "cpu") for m in self.five])
            p23 = np.mean([predict(m, xb, "cpu") for m in self.sub], 0)
        return dict(est=est, ok=est >= self.gate["threshold"], p5=p5n.mean(0), spread=p5n.std(0).mean(1), p23=p23)


def reliability_table(results_dir):
    """Internal test AUC of each named finding (Step 3.3c) -> tier used for display only."""
    path = os.path.join(results_dir, "sub23_leadII_auc.csv")
    if not os.path.exists(path):
        return {}
    t = pd.read_csv(path)
    out = {}
    for _, r in t.iterrows():
        tier = "high" if r["AUC"] >= 0.90 else ("moderate" if r["AUC"] >= 0.85 else "low")
        out[r["subclass"]] = dict(auc=float(r["AUC"]), ci=str(r["95% CI"]), n=int(r["test ECGs"]), tier=tier)
    return out


# ---------------------------------------------------------------- examples
def load_examples(path):
    if not os.path.exists(path):
        return [], {}
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return [(m["label"], m["key"]) for m in meta], {m["key"]: dict(m, signal=z[m["key"]] / 1000.0) for m in meta}


# ---------------------------------------------------------------- analysis
def analyze(sig_100, M, rel):
    n = min(len(sig_100) // (STRIP * FS), MAX_STRIPS)
    strips = np.stack([sig_100[i * STRIP * FS:(i + 1) * STRIP * FS] for i in range(n)])
    out = M.run(strips)
    flags5 = out["p5"] >= np.array([M.thr5[c] for c in SUPER])[None, :]
    flags23 = np.zeros_like(out["p23"], bool)
    for j, s in enumerate(M.sub_names):
        if s in M.thr23:
            flags23[:, j] = out["p23"][:, j] >= M.thr23[s]
    hr = []
    for x in strips:
        pk = detect_r_peaks(x, FS)
        rr = np.diff(pk) / FS
        rr = rr[(rr > 0.25) & (rr < 2.5)]
        hr.append(60 / rr.mean() if len(rr) else np.nan)
    out.update(strips=strips, n=n, flags5=flags5, flags23=flags23, hr=np.array(hr), k=0, rel=rel,
               names=M.sub_names, thr5=M.thr5, thr23=M.thr23, gate_thr=M.gate["threshold"])
    return out


def abnormal(res, i):
    return bool(res["ok"][i] and res["flags5"][i, 1:].any())


def findings(res, i, tiers=("high", "moderate")):
    """Named findings flagged in strip i, restricted to reliable tiers (abnormal subclasses only)."""
    out = []
    for j, s in enumerate(res["names"]):
        r = res["rel"].get(s)
        if s != "NORM" and res["flags23"][i, j] and r and r["tier"] in tiers:
            out.append((s, r))
    return sorted(out, key=lambda t: -t[1]["auc"])


# ---------------------------------------------------------------- figures
def ecg_paper(fig, a_s, b_s, ylim, title):
    xt = np.arange(np.floor(a_s / 0.2) * 0.2, b_s + 1e-9, 0.2)
    yt = np.arange(-ylim, ylim + 1e-9, 0.5)
    lab = lambda v: f"{v:.0f}" if abs(v - round(v)) < 1e-6 else ""
    fig.update_xaxes(range=[a_s, b_s], tickmode="array", tickvals=xt, ticktext=[lab(v) for v in xt],
                     showgrid=True, gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False, ticks="",
                     minor=dict(tickmode="linear", tick0=0, dtick=0.04, showgrid=True, gridcolor=GRID_MINOR, gridwidth=0.5),
                     title_text="time (s)  ·  grid: 0.2 s × 0.5 mV (small squares 0.04 s × 0.1 mV)")
    fig.update_yaxes(range=[-ylim, ylim], tickmode="array", tickvals=yt, ticktext=[lab(v) for v in yt],
                     showgrid=True, gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False, ticks="",
                     minor=dict(tickmode="linear", tick0=0, dtick=0.1, showgrid=True, gridcolor=GRID_MINOR, gridwidth=0.5),
                     title_text="mV")
    fig.update_layout(plot_bgcolor=PAPER, paper_bgcolor="white", height=360, margin=dict(l=55, r=20, t=50, b=45),
                      font=FONT, title=dict(text=title, font=dict(size=14)), showlegend=False)
    return fig


def ecg_fig(res, i, peaks=False):
    x = bandpass(res["strips"][i], FS)
    t = i * STRIP + np.arange(len(x)) / FS
    ylim = float(np.clip(np.ceil(np.percentile(np.abs(x), 99.9) / 0.5) * 0.5 + 0.5, 1.5, 4.0))
    fig = go.Figure(go.Scatter(x=t, y=x, mode="lines", line=dict(color="#111", width=1.3), hoverinfo="skip"))
    if peaks:
        p = detect_r_peaks(res["strips"][i], FS)
        fig.add_trace(go.Scatter(x=t[p], y=x[p], mode="markers", marker=dict(size=8, color="#e11d48"), hoverinfo="skip"))
    status = "too noisy to read" if not res["ok"][i] else ("findings" if abnormal(res, i) else "no abnormality flagged")
    return ecg_paper(fig, t[0], t[0] + STRIP, ylim, f"Lead II, strip {i + 1} of {res['n']} "
                     f"({mmss(i * STRIP)}–{mmss((i + 1) * STRIP)}): {status}")


def strip_bar(res):
    col = [C_NOISY if not res["ok"][i] else (C_BAD if abnormal(res, i) else C_OK) for i in range(res["n"])]
    fig = go.Figure()
    for i, c in enumerate(col):
        fig.add_shape(type="rect", x0=i * STRIP, x1=(i + 1) * STRIP - 0.4, y0=0, y1=1, fillcolor=c, line_width=0)
    k = res["k"]
    fig.add_shape(type="rect", x0=k * STRIP - 0.3, x1=(k + 1) * STRIP - 0.1, y0=-0.15, y1=1.15,
                  line=dict(color="#111", width=2), fillcolor="rgba(0,0,0,0)")
    for c, name in ((C_OK, "no abnormality flagged"), (C_BAD, "findings"), (C_NOISY, "too noisy to read")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=c, symbol="square"), name=name))
    fig.update_yaxes(range=[-0.3, 1.3], showticklabels=False, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_xaxes(range=[0, res["n"] * STRIP], title_text="time (s)", showgrid=False, zeroline=False)
    fig.update_layout(template="plotly_white", height=130, margin=dict(l=20, r=20, t=30, b=40), font=FONT,
                      hovermode=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def prob_fig(res, i):
    p, thr = 100 * res["p5"][i], [100 * res["thr5"][c] for c in SUPER]
    colors = [(C_NOISY if not res["ok"][i] else (C_BAD if (res["flags5"][i, j] and j) else
              (C_OK if (j == 0 and res["flags5"][i, 0]) else "#cbd5e1"))) for j in range(5)]
    fig = go.Figure(go.Bar(y=[PLAIN[c] for c in SUPER], x=p, orientation="h", marker_color=colors, showlegend=False,
                           text=[f"{v:.0f}%" for v in p], textposition="outside", hoverinfo="skip"))
    fig.add_trace(go.Scatter(y=[PLAIN[c] for c in SUPER], x=thr, mode="markers", name="threshold (chosen on validation)",
                             marker=dict(symbol="line-ns-open", size=22, color="#111", line=dict(width=2))))
    fig.update_xaxes(range=[0, 110], title_text="model probability (%)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=20, t=40, b=45), font=FONT,
                      title=dict(text=f"Five diagnostic groups, strip {i + 1}"
                                 + ("  ·  NOT RELIABLE: strip too noisy" if not res["ok"][i] else ""), font=dict(size=14)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"))
    return fig


# ---------------------------------------------------------------- cards and text
def quality_word(res, i):
    e = res["est"][i]
    if not res["ok"][i]:
        return "Too noisy", "bad", "estimated SNR below the gate"
    return ("Good", "ok", f"estimated SNR {e:.0f} dB") if e >= 15 else ("Fair", "warn", f"estimated SNR {e:.0f} dB")


def basic_cards(res, i):
    q, qs, qsub = quality_word(res, i)
    hr = res["hr"][i]
    items = [("signal", "Signal quality", q, qsub, qs),
             ("heart", "Heart rate", f"{hr:.0f} bpm" if np.isfinite(hr) else "—", "usual at rest: 60–100",
              "ok" if np.isfinite(hr) and 60 <= hr <= 100 else "warn")]
    if res["ok"][i]:
        groups = [PLAIN[c].split(" (")[0] for j, c in enumerate(SUPER) if j and res["flags5"][i, j]]
        items.append(("alert" if groups else "check", "Screening result",
                      f"{len(groups)} finding group(s)" if groups else "No abnormality flagged",
                      ", ".join(groups) if groups else "a 12-lead ECG is still needed to rule things out",
                      "bad" if groups else "ok"))
    if res["n"] > 1:
        nok = int(res["ok"].sum())
        nab = sum(abnormal(res, k) for k in range(res["n"]))
        items.append(("layers", "Strips", f"{nok} of {res['n']} readable", f"{nab} with findings", "ok" if nok == res["n"] else "warn"))
    return cards_html(items)


def findings_html(res, i):
    if not res["ok"][i]:
        return ('<div class="noisy"><b>This strip is too noisy to read.</b> No diagnosis is given for it. Check that the '
                'electrodes are firmly attached and that the person is still, then record again.</div>')
    groups = [(c, res["p5"][i, j]) for j, c in enumerate(SUPER) if j and res["flags5"][i, j]]
    named = findings(res, i)
    if not groups and not named:
        return ('<div class="find">No abnormality flagged in this strip. <span class="note">A single lead misses some '
                'diagnoses (for example many anterior infarctions and hypertrophy), so this does not rule them '
                'out.</span></div>')
    li = [f"<li><b>{PLAIN[c]}</b> <span class='note'>({100 * p:.0f}%)</span></li>" for c, p in groups]
    for s, r in named:
        cls = {"high": "b-high", "moderate": "b-mod"}[r["tier"]]
        li.append(f"<li>{SUBNAME.get(s, s)} <span class='badge {cls}'>{r['tier']} reliability · AUC {r['auc']:.2f}</span></li>")
    return ('<div class="find"><b>Possible findings</b> — please confirm with a 12-lead ECG and a doctor:<ul>'
            + "".join(li) + "</ul></div>")


def expert_html(ex):
    if not ex:
        return ""
    subs = [SUBNAME.get(s, s) for s in ex["subclass"]] or ["no diagnostic statement"]
    return cards_html([("usercheck", "Cardiologists' label (PTB-XL)", ", ".join(subs),
                        f"ECG {ex['ecg_id']} · test fold, never used for training" + (" · noise added" if ex.get("noise") else ""),
                        "")])


def summary_md(res, title, notes):
    lines = [f"**{title}**", f"{res['n']} strip(s) of 10 s analyzed; quality gate threshold: estimated SNR ≥ "
             f"{res['gate_thr']:.1f} dB."]
    nok = int(res["ok"].sum())
    lines.append(f"Readable strips: **{nok} of {res['n']}**.")
    if nok:
        rows = [f"{PLAIN[c]}: flagged in {int(res['flags5'][res['ok'], j].sum())} of {nok}" for j, c in enumerate(SUPER) if j]
        lines.append("; ".join(rows) + ".")
    if notes:
        lines.append("*" + "; ".join(notes) + ".*")
    return "\n\n".join(lines)


def named_table(res, i):
    rows = []
    for j, s in enumerate(res["names"]):
        r = res["rel"].get(s, {})
        rows.append({"finding": SUBNAME.get(s, s), "code": s, "probability %": round(float(100 * res["p23"][i, j]), 1),
                     "threshold %": round(float(100 * res["thr23"][s]), 1) if s in res["thr23"] else None,
                     "flagged": ("yes" if res["flags23"][i, j] else "") if res["ok"][i] else "(noisy strip)",
                     "reliability": r.get("tier", "not measured"),
                     "test AUC (95% CI)": f"{r['auc']:.3f} ({r['ci']})" if r else "", "test ECGs": r.get("n")})
    return pd.DataFrame(rows).sort_values("probability %", ascending=False)


def quality_md(res, i):
    return (f"Estimated SNR **{res['est'][i]:.1f} dB** (gate: ≥ {res['gate_thr']:.1f} dB) → "
            f"**{'readable' if res['ok'][i] else 'too noisy, not diagnosed'}**. Disagreement between the 5 networks: "
            f"{100 * res['spread'][i]:.1f} percentage points (mean over the 5 groups).")


def strips_table(res):
    rows = []
    for i in range(res["n"]):
        r = {"strip": i + 1, "start": mmss(i * STRIP), "estimated SNR dB": round(float(res["est"][i]), 1),
             "readable": bool(res["ok"][i]), "heart rate": round(float(res["hr"][i]), 0) if np.isfinite(res["hr"][i]) else None}
        r.update({f"{c} %": round(float(100 * res["p5"][i, j])) for j, c in enumerate(SUPER)})
        r["named findings"] = ", ".join(s for s, _ in findings(res, i)) if res["ok"][i] else ""
        rows.append(r)
    df = pd.DataFrame(rows)
    f = tempfile.NamedTemporaryFile(delete=False, suffix="_strips.csv")
    df.to_csv(f.name, index=False)
    return df, f.name


# ---------------------------------------------------------------- app
def build_demo(models_dir, examples_path, results_dir, about_md):
    M = Models(models_dir)
    rel = reliability_table(results_dir)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    choices, EX = load_examples(examples_path)

    def run(source, ex_key, upload, column, fs):
        notes, ex = [], None
        if source == "Upload my ECG":
            if not upload:
                raise gr.Error("Upload a CSV file first, or switch to 'Example'.")
            fs = float(fs or 0)
            if fs < 100:
                raise gr.Error("Please enter the sampling rate of your file (at least 100 Hz).")
            sig = read_signal(upload, column)
            if len(sig) < STRIP * fs:
                raise gr.Error("The recording is shorter than 10 seconds.")
            if len(sig) > MAX_STRIPS * STRIP * fs:
                gr.Warning(f"Longer than {MAX_STRIPS * STRIP // 60} min: only the first {MAX_STRIPS * STRIP // 60} min are analyzed.")
                sig = sig[: int(MAX_STRIPS * STRIP * fs)]
            sig = to_100hz(sig, fs)
            sig, notes = check_amplitude(sig)
            title = f"Your upload · {len(sig) / FS:.0f} s at {fs:g} Hz · column '{column}'"
        else:
            ex = EX[ex_key]
            sig, title = ex["signal"], ex["label"]
        res = analyze(sig, M, rel)
        res.update(title=title, notes=notes, ex=ex)
        first = [i for i in range(res["n"]) if abnormal(res, i)]
        res["k"] = first[0] if first else 0
        return res

    def views(res):
        k = res["k"]
        return (basic_cards(res, k), findings_html(res, k), gr.update(value=strip_bar(res), visible=res["n"] > 1),
                ecg_fig(res, k),
                prob_fig(res, k), named_table(res, k), quality_md(res, k), ecg_fig(res, k, peaks=True), res)

    def render(res):
        tbl, csv = strips_table(res)
        return views(res) + (expert_html(res["ex"]), summary_md(res, res["title"], res["notes"]), tbl, csv)

    def shift(res, step):
        if res is None:
            raise gr.Error("Analyze an ECG first.")
        res["k"] = int(np.clip(res["k"] + step, 0, res["n"] - 1))
        return views(res)

    def next_finding(res):
        if res is None:
            raise gr.Error("Analyze an ECG first.")
        nxt = [i for i in range(res["n"]) if abnormal(res, i) and i > res["k"]] or \
              [i for i in range(res["n"]) if abnormal(res, i)]
        if not nxt:
            gr.Info("No strip with findings.")
            return views(res)
        res["k"] = nxt[0]
        return views(res)

    def on_upload(f):
        if not f:
            return gr.update(choices=[], value=None)
        names, _ = read_columns(f)
        return gr.update(choices=names, value=default_column(names))

    theme = gr.themes.Soft(primary_hue="rose", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kw, launch_kw = dict(theme=theme, css=CSS), {}
    else:
        blocks_kw, launch_kw = {}, dict(theme=theme, css=CSS)
    default = choices[0][1] if choices else None

    with gr.Blocks(title="ECG Lead II Diagnosis", **blocks_kw) as demo:
        state = gr.State()
        gr.HTML('<div class="title">ECG Lead II Diagnosis</div>')
        with gr.Group():
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    gr.HTML('<div class="step">1 · Choose an ECG</div>')
                    source = gr.Radio(["Example", "Upload my ECG"], value="Example", show_label=False)
                    example = gr.Dropdown(choices, value=default, label="Example (PTB-XL test fold, never used for training)")
                    with gr.Row(visible=False) as up_row:
                        upload = gr.File(label="Your ECG file (.csv)", file_types=[".csv", ".txt"], type="filepath", scale=2)
                        with gr.Column(scale=1):
                            column = gr.Dropdown([], label="ECG column (lead II)", allow_custom_value=True)
                            fs = gr.Number(500, label="Sampling rate (Hz)", precision=0)
                with gr.Column(scale=1):
                    gr.HTML('<div class="step">2 · Analyze</div>')
                    run_btn = gr.Button("Analyze ECG", variant="primary", size="lg")
        with gr.Tabs():
            with gr.Tab("Basic"):
                b_cards = gr.HTML()
                b_find = gr.HTML()
                b_bar = gr.Plot(show_label=False)
                b_ecg = gr.Plot(show_label=False)
                with gr.Row():
                    b_prev = gr.Button("◀ previous strip", size="sm")
                    b_next = gr.Button("next strip ▶", size="sm")
                    b_nf = gr.Button("Next strip with findings ⏭", size="sm", variant="secondary")
                b_exp = gr.HTML()
                gr.Markdown("<small>Research and education tool, not a medical device. A single lead cannot replace a "
                            "12-lead ECG; it can only point to who needs one.</small>")
            with gr.Tab("Advanced"):
                gr.HTML(sec("Recording"))
                a_summary = gr.Markdown()
                gr.HTML(sec("Five diagnostic groups (this strip)"))
                a_prob = gr.Plot(show_label=False)
                gr.HTML(sec("Named findings (23 PTB-XL subclasses, this strip)"))
                gr.Markdown("<small>Reliability = how well lead II alone detected the finding on the PTB-XL test fold "
                            "(high: AUC ≥ 0.90; moderate: 0.85–0.90; low: below 0.85). Only high and moderate findings "
                            "are shown on the Basic tab.</small>")
                a_named = gr.Dataframe(interactive=False)
                gr.HTML(sec("Signal quality"))
                a_q = gr.Markdown()
                a_ecg = gr.Plot(show_label=False)
                with gr.Row():
                    a_prev = gr.Button("◀ previous strip", size="sm")
                    a_next = gr.Button("next strip ▶", size="sm")
                gr.HTML(sec("All strips"))
                a_tbl = gr.Dataframe(interactive=False)
                a_csv = gr.File(label="All strips (CSV)")
            with gr.Tab("How it works"):
                gr.Markdown(DESC)
                gr.HTML(REQ_HTML)
                gr.HTML(PIPE_HTML)
                gr.Markdown(INPUT_MD)
                gr.Markdown(about_md)
        gr.Markdown("<center><small>Fady Mostafa Omar · 2026 · Data: PTB-XL (Wagner et al., PhysioNet, CC BY 4.0), "
                    "MIT-BIH Noise Stress Test Database, Chapman-Shaoxing (PhysioNet)</small></center>")

        nav = [b_cards, b_find, b_bar, b_ecg, a_prob, a_named, a_q, a_ecg, state]
        outs = nav + [b_exp, a_summary, a_tbl, a_csv]
        source.change(lambda s: (gr.update(visible=s == "Example"), gr.update(visible=s != "Example")),
                      source, [example, up_row])
        upload.change(on_upload, upload, column)
        run_btn.click(lambda s, e, u, c, f: render(run(s, e, u, c, f)), [source, example, upload, column, fs], outs)
        example.change(lambda e: render(run("Example", e, None, None, None)), example, outs)
        if default is not None:
            demo.load(lambda: render(run("Example", default, None, None, None)), None, outs)
        for btn, fn in ((b_prev, lambda r: shift(r, -1)), (b_next, lambda r: shift(r, 1)),
                        (a_prev, lambda r: shift(r, -1)), (a_next, lambda r: shift(r, 1)), (b_nf, next_finding)):
            btn.click(fn, state, nav)
    return demo, launch_kw


def build_about(results_dir):
    """'How it works' text, filled from the result files of Steps 3.3-3.6."""
    R = lambda f: os.path.join(results_dir, f)
    parts = []
    if os.path.exists(R("cnn_auc.csv")):
        t = pd.read_csv(R("cnn_auc.csv")).astype(object).fillna("")
        rows = "\n".join(f"| {r['config']} | {r['model']} | " + " | ".join(str(r[c]) for c in SUPER) + f" | {r['macro AUC']} |"
                         for _, r in t.iterrows())
        parts.append("### Accuracy on the locked PTB-XL test fold (2,158 ECGs, patients never seen in training)\n"
                     "| Input | Model | " + " | ".join(SUPER) + " | Macro AUC |\n|---|---|" + "---|" * 6 + "\n" + rows)
    if os.path.exists(R("noise_macro_auc_table.csv")):
        t = pd.read_csv(R("noise_macro_auc_table.csv"))
        t = t[(t["lead"].astype(str) == "II") & (t["noise"] == "all three mixed")]
        cols = [c for c in t.columns if c.endswith("dB") or c == "no noise"]
        rows = "\n".join(f"| {r['model']} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |" for _, r in t.iterrows())
        parts.append("### Real recorded noise (MIT-BIH Noise Stress Test Database; training and test noise from "
                     "different parts of each recording)\nLead II, macro AUC, baseline wander + electrode motion + "
                     "muscle artefact mixed:\n\n| Model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n" + rows)
    if os.path.exists(R("quality_gate_rejections.csv")) and os.path.exists(R("quality_gate_auc.csv")):
        rj = pd.read_csv(R("quality_gate_rejections.csv")).iloc[0]
        ga = pd.read_csv(R("quality_gate_auc.csv")).iloc[0]
        parts.append("### Quality gate\nOn noisy test strips it refused "
                     f"{rj['sensitivity (SNR < 0 dB rejected)']} of the strips with SNR below 0 dB and "
                     f"{rj['false rejections (SNR >= 0 dB)']} of the others. Macro AUC: all strips {ga['all strips']}, "
                     f"accepted strips {ga['accepted']}, refused strips {ga['refused']}.")
    if os.path.exists(R("sub23_leadII_auc.csv")):
        t = pd.read_csv(R("sub23_leadII_auc.csv"))
        rows = "\n".join(f"| {SUBNAME.get(r['subclass'], r['subclass'])} | {int(r['test ECGs'])} | {r['AUC']:.3f} | {r['95% CI']} |"
                         for _, r in t.sort_values("AUC", ascending=False).iterrows())
        parts.append("### Named findings from lead II alone (test fold, one-vs-rest)\n| Finding | Test ECGs | AUC | 95% CI |\n"
                     "|---|---|---|---|\n" + rows)
    if os.path.exists(R("thresholds_test_performance.csv")):
        t = pd.read_csv(R("thresholds_test_performance.csv"))
        t = t[t["kind"] == "group"]
        rows = "\n".join(f"| {PLAIN.get(r['name'], r['name'])} | {r['threshold %']:.0f}% | {r['sensitivity %']:.0f}% | "
                         f"{r['specificity %']:.0f}% | {r['PPV %']:.0f}% |" for _, r in t.iterrows())
        parts.append("### Decisions (thresholds chosen on the validation fold, scored on the test fold)\n"
                     "| Group | Threshold | Sensitivity | Specificity | PPV |\n|---|---|---|---|---|\n" + rows)
    full, part = R("external_chapman_auc_full.csv"), R("external_chapman_auc.csv")
    if os.path.exists(full):
        t = pd.read_csv(full)
        t = t[(t["mapping"] == "PTB-XL-consistent") & t["model"].isin(["lead II, noise-trained", "lead II, 23 subclasses"])]
        rows = "\n".join(f"| {r['task']} | {r['AUC external']:.3f} | {r['AUC internal']:.3f} |" for _, r in t.iterrows())
        parts.append("### External test: Chapman-Shaoxing (China), all ECGs, no retraining\n| Group / finding | Chapman AUC | "
                     "PTB-XL AUC |\n|---|---|---|\n" + rows)
    elif os.path.exists(part):
        t = pd.read_csv(part)
        t = t[t["model"].isin(["lead II, noise-trained", "lead II, 23 subclasses"])]
        rows = "\n".join(f"| {r['task']} | {r['AUC external']:.3f} | {r['AUC internal (PTB-XL test)']:.3f} |"
                         for _, r in t.iterrows())
        parts.append("### External test: Chapman-Shaoxing (China), 10,000 random ECGs, no retraining\n*Preliminary: the "
                     "pre-specified label mapping missed several codes (see the README).*\n\n| Group / finding | Chapman AUC | "
                     "PTB-XL AUC |\n|---|---|---|\n" + rows)
    parts.append("""### Limitations
- One lead cannot see everything: anterior and lateral infarction, incomplete RBBB and hypertrophy are detected much
  less well than with 12 leads. "No abnormality flagged" does not rule them out.
- The noise tests add real recorded noise to clean hospital ECGs; recordings made with noise are not identical.
- PTB-XL and Chapman-Shaoxing use different label systems, so the external numbers are approximate.
- AD8232 and other low-cost boards have not been tested yet.
- Research and education tool, not a medical device.""")
    return "\n\n".join(parts)


def main():
    import argparse
    root = os.path.join(HERE, "..")
    p = argparse.ArgumentParser(description="ECG Lead II Diagnosis (Gradio app)")
    p.add_argument("--models", default=os.path.join(root, "models"))
    p.add_argument("--examples", default=os.path.join(HERE, "examples", "examples.npz"))
    p.add_argument("--results", default=os.path.join(root, "results"))
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    a = p.parse_args()
    demo, launch_kw = build_demo(a.models, a.examples, a.results, build_about(a.results))
    demo.launch(share=a.share, server_port=a.port, **launch_kw)


if __name__ == "__main__":
    main()
