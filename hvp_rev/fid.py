"""Frechet Inception Distance (per-frame) for the renderer gate. InceptionV3 (ImageNet) pool features.

MMNIST is out-of-domain for ImageNet-Inception, so treat FID here as a RELATIVE distribution distance for
RANKING arms (lower = predicted-frame distribution closer to GT-frame distribution), never an absolute score.
64x64 gray frames -> resize 299 + replicate to 3ch (standard). Reusable across the gate.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn.functional as F
from scipy import linalg

_inc = None
def _model(dev):
    global _inc
    if _inc is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        m.fc = torch.nn.Identity(); m.eval()
        _inc = m.to(dev)
    return _inc

@torch.no_grad()
def features(frames, dev, bs=128):
    """frames: (N,1,64,64) or (N,64,64) tensor/array in [0,1] -> (N,2048) inception features."""
    if not torch.is_tensor(frames): frames = torch.as_tensor(frames)
    frames = frames.float()
    if frames.dim() == 3: frames = frames.unsqueeze(1)
    m = _model(dev); out = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    for i in range(0, frames.shape[0], bs):
        x = frames[i:i+bs].to(dev).clamp(0, 1).repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        out.append(m(x).cpu())
    return torch.cat(out).numpy()

def _stats(feat): return feat.mean(0), np.cov(feat, rowvar=False)

def frechet(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((s1 + eps*np.eye(s1.shape[0])).dot(s2 + eps*np.eye(s2.shape[0])))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2*np.trace(covmean))

def fid(pred_frames, gt_frames, dev):
    """pred/gt: (N,1,64,64) in [0,1]. Returns scalar FID (relative ranking metric)."""
    fp, fg = features(pred_frames, dev), features(gt_frames, dev)
    return frechet(*_stats(fp), *_stats(fg))
