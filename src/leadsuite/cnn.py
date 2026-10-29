"""Residual 1D-CNN for the 10-s PTB-XL ECGs (100 Hz) with any number of leads; multi-label output (5 superclasses).
Signals are band-passed 0.5-40 Hz and kept in mV (scaled by one global factor per lead), so amplitude information
(hypertrophy) is not normalized away. No polarity flips: polarity carries diagnostic information."""
import contextlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.signal import butter, sosfiltfilt

from .metrics import macro_auc


def preprocess(X, gain, fs=100, lo=0.5, hi=40.0):
    """int16 [n, samples, leads] -> float32 [n, leads, samples] in mV, band-passed."""
    sos = butter(3, [lo, hi], btype="band", fs=fs, output="sos")
    out = np.empty((X.shape[0], X.shape[2], X.shape[1]), np.float32)
    for i in range(0, len(X), 2000):
        out[i:i + 2000] = sosfiltfilt(sos, X[i:i + 2000].astype(np.float32) / gain, axis=1).transpose(0, 2, 1)
    return out


class ResBlock(nn.Module):
    def __init__(self, cin, cout, k=9, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        return Fn.relu(self.b2(self.c2(Fn.relu(self.b1(self.c1(x))))) + self.skip(x))


class LeadNet(nn.Module):
    def __init__(self, n_leads, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_leads, 64, 15, 1, 7, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        plan = [(64, 64, 2), (64, 128, 2), (128, 128, 2), (128, 192, 2), (192, 192, 2)]
        self.blocks = nn.Sequential(*[ResBlock(a, b, 9, s) for a, b, s in plan])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(192 * 2, n_classes))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


def _amp(device):
    """Mixed precision on a GPU (about 2-3x faster on a T4); plain float32 on a CPU."""
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict(model, X, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], device=device)
        with _amp(device):
            logits = model(xb)
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


def train(X, Y, tr, va, epochs=30, batch=128, lr=1e-3, seed=0, device="cpu", log=print, aug=None, val=None):
    """Train on rows tr; keep the epoch with the best validation macro AUC on rows va.
    aug: optional extra augmentation applied to every training batch (Step 3.4: real recorded noise).
    val: optional (Xv, Yv) used for choosing the epoch instead of X[va], Y[va]."""
    torch.manual_seed(seed); np.random.seed(seed)
    gpu = str(device).startswith("cuda")
    torch.backends.cudnn.benchmark = gpu
    scaler = torch.amp.GradScaler("cuda", enabled=gpu)
    Xt = torch.as_tensor(X[tr], device=device)
    Yt = torch.as_tensor(Y[tr], dtype=torch.float32, device=device)
    model = LeadNet(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(Yt) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss()
    L = X.shape[2]
    Xv, Yv = val if val is not None else (X[va], Y[va])
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Yt), device=device)
        for i in range(0, len(Yt), batch):
            idx = perm[i:i + batch]
            xb = Xt[idx] * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, device=device))         # amplitude
            shift = int(np.random.randint(-50, 51))                                         # up to +-0.5 s
            if shift:
                xb = torch.roll(xb, shift, dims=2)
                if shift > 0:
                    xb[:, :, :shift] = 0
                else:
                    xb[:, :, L + shift:] = 0
            xb = xb + 0.02 * torch.randn_like(xb)                                           # small noise
            if aug is not None:
                xb = aug(xb)
            with _amp(device):
                loss = loss_fn(model(xb).float(), Yt[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        score = macro_auc(Yv, predict(model, Xv, device))
        log(f"epoch {ep:2d} | loss {loss.item():.3f} | validation macro AUC {score:.3f} | {time.time() - t0:.0f} s")
        if score > best:
            best, best_ep = score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu().eval(), best, best_ep


def save_members(path, members, meta):
    ck = {"members": [m.state_dict() for m in members], "n_leads": members[0].stem[0].in_channels,
          "n_classes": members[0].head[1].out_features}
    ck.update(meta)
    torch.save(ck, path)


def load_members(path):
    ck = torch.load(path, map_location="cpu")
    out = []
    for sd in ck["members"]:
        m = LeadNet(ck["n_leads"], ck.get("n_classes", 5))
        m.load_state_dict(sd)
        out.append(m.eval())
    return out, ck
