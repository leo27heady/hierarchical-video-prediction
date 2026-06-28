"""GAN modules for the Step-1 adversarial renderer: a fresh renderer decoder (code -> frame), a pix2pix
PatchGAN discriminator (70x70 receptive field), and VQGAN's adaptive-lambda weighting (the C0b texture guard).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class RendererDecoder(nn.Module):
    """Fresh decoder mirroring CW-VAE's ConvDecoder: code (220) -> Dense(1024) -> 4x ConvTranspose -> (1,64,64).
    sigmoid output to match [0,1] targets (cleaner than tanh for the renderer)."""
    def __init__(self, feat_in=220, ch_out=1, cm=1):
        super().__init__()
        c = 32 * cm
        self.dense = nn.Linear(feat_in, 1024 * cm)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024 * cm, 4*c, 5, 2), nn.LeakyReLU(0.2),   # 1->5
            nn.ConvTranspose2d(4*c, 2*c, 5, 2), nn.LeakyReLU(0.2),         # 5->13
            nn.ConvTranspose2d(2*c, c, 6, 2), nn.LeakyReLU(0.2),           # 13->30
            nn.ConvTranspose2d(c, ch_out, 6, 2),                          # 30->64
        )
    def last_layer(self):
        return self.net[-1].weight                                        # for adaptive-lambda grad norm
    def forward(self, z):
        h = self.dense(z).view(z.shape[0], -1, 1, 1)
        return torch.sigmoid(self.net(h))


class PatchGAN(nn.Module):
    """pix2pix NLayerDiscriminator (n_layers=3 -> ~70x70 receptive field). Unconditional on a (1,64,64) frame.
    Optional spectral norm for stability. Outputs a patch map of real/fake logits."""
    def __init__(self, in_ch=1, ndf=64, n_layers=3, spectral=True):
        super().__init__()
        def sn(m): return nn.utils.spectral_norm(m) if spectral else m
        layers = [sn(nn.Conv2d(in_ch, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev, nf = nf, min(ndf * 2**n, 512)
            layers += [sn(nn.Conv2d(nf_prev, nf, 4, 2, 1)), nn.BatchNorm2d(nf) if not spectral else nn.Identity(),
                       nn.LeakyReLU(0.2, True)]
        nf_prev, nf = nf, min(ndf * 2**n_layers, 512)
        layers += [sn(nn.Conv2d(nf_prev, nf, 4, 1, 1)), nn.BatchNorm2d(nf) if not spectral else nn.Identity(),
                   nn.LeakyReLU(0.2, True), sn(nn.Conv2d(nf, 1, 4, 1, 1))]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def hinge_d_loss(real_logits, fake_logits):
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())

def hinge_g_loss(fake_logits):
    return -fake_logits.mean()


def adaptive_lambda(nll_loss, adv_loss, last_layer, max_lam=1e4, eps=1e-6):
    """VQGAN adaptive weight: lambda = ||grad_lastlayer(nll)|| / (||grad_lastlayer(adv)|| + eps), detached.
    Auto-clamps the GAN term so it cannot run away into incoherent texture (the C0b guard)."""
    nll_grad = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    adv_grad = torch.autograd.grad(adv_loss, last_layer, retain_graph=True)[0]
    lam = nll_grad.norm() / (adv_grad.norm() + eps)
    return lam.clamp(0, max_lam).detach()
