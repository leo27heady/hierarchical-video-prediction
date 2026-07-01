import torch.nn as nn
class _HParams(dict):
    def update(self, other):
        super().update(other)
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
class LightningModule(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.hparams = _HParams()
