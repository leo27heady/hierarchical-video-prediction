"""Phase-C factor metrics for standard-MMNIST long rollouts (pre-registered METRIC MANDATE,
SPATIAL_FACTORIAL.md 2026-07-10): beyond fg-MSE, CW-VAE-facing retention needs
  (i)  digit-IDENTITY retention — a fixed MNIST classifier reads each predicted digit; retention
       = agreement with the classes anchored on the last GT context frame (position-matched);
  (ii) TRAJECTORY error — centroid distance between predicted digits and the GT frame's digits
       (nearest-match assignment; the generator is deterministic so GT exists at every horizon);
  (iii) digit COUNT deviation — |#detected − #GT-detected| per frame.
Aggregate MSE/SSIM alone misses exactly the "identity persists while position degrades"
distinction CW-VAE emphasizes — hence this module.

Digits are localized by connected components on thresholded frames (MMNIST: bright strokes on
black; default thr 0.3), cropped around the component bounding box (square, padded), resized to
28x28 for the classifier. Overlapping digits merge into one component — counted as a detection
of 1 and matched to the nearest GT centroid; identity is scored only on frames where detected
count == anchor count (disclosed as `id_scored` fraction).

  # train the classifier once (~2 min):
  venv/Scripts/python.exe -m hvp_rev.mmnist_metrics --train_classifier
  # CPU smoke (synthetic blobs; no GPU, no files):
  CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m hvp_rev.mmnist_metrics --smoke
"""
from __future__ import annotations

import argparse
import gzip
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MNIST_IDX = "C:/Users/leoni/Documents/projects/OpenSTL/data/moving_mnist/train-images-idx3-ubyte.gz"
MNIST_LBL = MNIST_IDX.replace("images-idx3", "labels-idx1")
CKPT = "logs_rhcp/mmnist_digit_clf.pt"
THR = 0.3
MIN_PIX = 30            # ignore specks below this component size


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def train_classifier(steps=3000, batch=256, lr=1e-3):
    with gzip.open(MNIST_IDX, "rb") as f:
        imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(MNIST_LBL, "rb") as f:
        lbls = np.frombuffer(f.read(), np.uint8, offset=8)
    x = torch.from_numpy(imgs.copy()).float().unsqueeze(1) / 255.0
    y = torch.from_numpy(lbls.astype(np.int64))
    n_val = 5000
    xt, yt, xv, yv = x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]
    net = DigitCNN().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for s in range(1, steps + 1):
        i = torch.randint(0, len(xt), (batch,))
        loss = F.cross_entropy(net(xt[i].to(DEV)), yt[i].to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % 500 == 0:
            with torch.no_grad():
                acc = (net(xv.to(DEV)).argmax(1).cpu() == yv).float().mean()
            print(f"  [{s}/{steps}] loss {float(loss):.4f}  val acc {float(acc):.4f}", flush=True)
    torch.save(net.state_dict(), CKPT)
    print(f"saved {CKPT}")
    return net


def load_classifier():
    net = DigitCNN().to(DEV)
    net.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True))
    return net.eval()


def detect_digits(frame01, thr=THR):
    """frame01: (H,W) numpy in [0,1] -> list of (centroid_yx, crop28) sorted by x. Components
    below MIN_PIX are ignored (spawn specks / decode noise)."""
    mask = frame01 > thr
    lab, n = ndimage.label(mask)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) < MIN_PIX:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        h = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        half = max(h // 2 + 2, 14)
        y0, x0 = int(round(cy)), int(round(cx))
        pad = np.zeros((2 * half, 2 * half), np.float32)
        ys0, ys1 = max(0, y0 - half), min(frame01.shape[0], y0 + half)
        xs0, xs1 = max(0, x0 - half), min(frame01.shape[1], x0 + half)
        pad[ys0 - (y0 - half):ys0 - (y0 - half) + (ys1 - ys0),
            xs0 - (x0 - half):xs0 - (x0 - half) + (xs1 - xs0)] = frame01[ys0:ys1, xs0:xs1]
        crop = torch.from_numpy(pad)[None, None]
        crop28 = F.interpolate(crop, size=(28, 28), mode="bilinear", align_corners=False)[0]
        out.append(((cy, cx), crop28))
    return sorted(out, key=lambda t: t[0][1])


def classify(net, crops):
    if not crops:
        return []
    x = torch.stack([c for _, c in crops]).to(DEV)
    with torch.no_grad():
        return net(x).argmax(1).cpu().tolist()


def frame_metrics(net, pred01, gt01, anchor_classes, anchor_used):
    """One rollout frame -> dict(count_dev, traj_err, id_match, id_scored).
    traj_err: mean nearest-centroid distance pred->GT detections (px; NaN if either empty).
    id: predicted digits classified and position-matched to GT detections; scored only when
    detected count == len(anchor_classes) (disclosed via id_scored)."""
    pd = detect_digits(pred01)
    gd = detect_digits(gt01)
    count_dev = abs(len(pd) - len(gd))
    if pd and gd:
        pc = np.array([c for c, _ in pd]); gc = np.array([c for c, _ in gd])
        d = np.linalg.norm(pc[:, None, :] - gc[None, :, :], axis=-1)
        traj = float(d.min(axis=1).mean())
    else:
        traj = float("nan")
    id_match, id_scored = float("nan"), 0.0
    if anchor_used and len(pd) == len(anchor_classes) and pd:
        cls = classify(net, pd)
        # anchors and detections both sorted by x at anchor time; match by nearest GT centroid
        # order to keep assignment stable under crossings is out of scope (disclosed)
        id_match = float(np.mean([c == a for c, a in zip(cls, anchor_classes)]))
        id_scored = 1.0
    return dict(count_dev=count_dev, traj_err=traj, id_match=id_match, id_scored=id_scored)


def _smoke():
    torch.manual_seed(0)
    net = DigitCNN()             # untrained: smoke checks plumbing, not accuracy
    f = np.zeros((64, 64), np.float32)
    f[10:24, 8:20] = 0.9         # blob 1
    f[40:56, 40:52] = 0.8        # blob 2
    det = detect_digits(f)
    assert len(det) == 2, len(det)
    assert det[0][1].shape == (1, 28, 28)
    cls = classify(net, det)
    assert len(cls) == 2
    m = frame_metrics(net, f, f, anchor_classes=cls, anchor_used=True)
    assert m["count_dev"] == 0 and m["traj_err"] < 1e-6 and m["id_scored"] == 1.0
    assert m["id_match"] == 1.0  # same frame, same (untrained) classifier -> classes agree
    empty = np.zeros((64, 64), np.float32)
    m2 = frame_metrics(net, empty, f, anchor_classes=cls, anchor_used=True)
    assert m2["count_dev"] == 2 and np.isnan(m2["traj_err"]) and m2["id_scored"] == 0.0
    print("mmnist_metrics SMOKE OK (detection, cropping, classify, matching, empty-frame path)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_classifier", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        _smoke()
    elif a.train_classifier:
        train_classifier()
    else:
        ap.print_help()
