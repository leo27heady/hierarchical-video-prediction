"""OpenSTL Lightning wrapper for the integrated hierarchical predictor."""
from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
import torch
from skimage.metrics import structural_similarity

from openstl.models import HVPIntegrated_Model
from .base_method import Base_method


class HVPIntegrated(Base_method):
    """Hierarchical allocation, dynamic routing, and reversible computation."""

    def _build_model(self, **args):
        return HVPIntegrated_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        del batch_y
        return self.model(batch_x, **kwargs)

    def _shared_step(self, batch, stage: str):
        batch_x, batch_y = batch
        prediction, auxiliary = self.model(batch_x, return_aux=True)
        loss = self.criterion(prediction, batch_y)
        batch_size = batch_x.shape[0]
        self.log(
            f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True,
            prog_bar=stage != "test", batch_size=batch_size,
        )
        frame_mse = (prediction - batch_y).square().sum(dim=(-3, -2, -1)).mean()
        pixel_mse = (prediction - batch_y).square().mean()
        pixel_mae = (prediction - batch_y).abs().mean()
        self.log(
            f"{stage}_frame_mse", frame_mse, on_step=False, on_epoch=True,
            prog_bar=False, batch_size=batch_size,
        )
        self.log(
            f"{stage}_pixel_mse", pixel_mse, on_step=False, on_epoch=True,
            prog_bar=False, batch_size=batch_size,
        )
        self.log(
            f"{stage}_pixel_mae", pixel_mae, on_step=False, on_epoch=True,
            prog_bar=False, batch_size=batch_size,
        )
        for name in (
            "route_entropy",
            "route_temporal_coverage",
            "route_global_unconstrained_score_gap",
            "route_churn",
            "route_spatial_coverage",
            "route_frame_std",
            "route_selected",
            "packed_tokens",
        ):
            value = auxiliary[name]
            if torch.isfinite(value).all():
                self.log(
                    f"{stage}_{name}", value, on_step=False, on_epoch=True,
                    prog_bar=False, batch_size=batch_size,
                )
        return loss, prediction, auxiliary

    def training_step(self, batch, batch_idx):
        del batch_idx
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, prediction, auxiliary = self._shared_step(batch, "val")
        if batch_idx == 0:
            batch_x, batch_y = batch
            self.validation_panel = {
                "inputs": batch_x[:4].detach().cpu(),
                "targets": batch_y[:4].detach().cpu(),
                "predictions": prediction[:4].detach().cpu(),
                "route_indices": auxiliary["route_indices"][:4].detach().cpu(),
            }
        return loss

    def on_before_optimizer_step(self, optimizer):
        del optimizer
        interval = int(self.hparams.get("evidence_log_every", 50))
        if interval <= 0 or self.global_step % interval:
            return
        squared = torch.zeros((), device=self.device)
        parameter_squared = torch.zeros((), device=self.device)
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter_squared = parameter_squared + parameter.detach().float().square().sum()
            if parameter.grad is not None:
                squared = squared + parameter.grad.detach().float().square().sum()
        self.log("train_grad_norm", squared.sqrt(), on_step=True, on_epoch=False)
        self.log("train_parameter_norm", parameter_squared.sqrt(), on_step=True, on_epoch=False)

    def on_test_start(self):
        horizons = int(self.hparams.aft_seq_length)
        self._test_sums = {
            "squared": torch.zeros((), device=self.device, dtype=torch.float64),
            "absolute": torch.zeros((), device=self.device, dtype=torch.float64),
            "pixels": torch.zeros((), device=self.device, dtype=torch.float64),
            "frames": torch.zeros((), device=self.device, dtype=torch.float64),
            "psnr": torch.zeros((), device=self.device, dtype=torch.float64),
            "ssim": torch.zeros((), device=self.device, dtype=torch.float64),
            "horizon_squared": torch.zeros(horizons, device=self.device, dtype=torch.float64),
            "horizon_absolute": torch.zeros(horizons, device=self.device, dtype=torch.float64),
            "horizon_psnr": torch.zeros(horizons, device=self.device, dtype=torch.float64),
            "horizon_ssim": torch.zeros(horizons, device=self.device, dtype=torch.float64),
            "samples": torch.zeros((), device=self.device, dtype=torch.float64),
        }
        self.test_panel = None

    @staticmethod
    def _frame_ssim(prediction, target):
        prediction = prediction.detach().float().clamp(0, 1).cpu().numpy()
        target = target.detach().float().clamp(0, 1).cpu().numpy()
        values = np.empty(prediction.shape[:2], dtype=np.float64)
        for batch_index in range(prediction.shape[0]):
            for time_index in range(prediction.shape[1]):
                pred_frame = prediction[batch_index, time_index]
                true_frame = target[batch_index, time_index]
                if pred_frame.shape[0] == 1:
                    pred_frame = pred_frame[0]
                    true_frame = true_frame[0]
                    channel_axis = None
                else:
                    pred_frame = np.moveaxis(pred_frame, 0, -1)
                    true_frame = np.moveaxis(true_frame, 0, -1)
                    channel_axis = -1
                values[batch_index, time_index] = structural_similarity(
                    pred_frame, true_frame, data_range=1.0, channel_axis=channel_axis
                )
        return values

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        prediction = self(batch_x)
        error = prediction.double() - batch_y.double()
        clipped_error = prediction.double().clamp(0, 1) - batch_y.double().clamp(0, 1)
        per_frame_pixel_mse = clipped_error.square().mean(dim=(-3, -2, -1)).clamp_min(1e-12)
        per_frame_psnr = -10.0 * torch.log10(per_frame_pixel_mse)
        frame_ssim = torch.as_tensor(
            self._frame_ssim(prediction, batch_y), device=self.device, dtype=torch.float64
        )
        self._test_sums["squared"] += error.square().sum()
        self._test_sums["absolute"] += error.abs().sum()
        self._test_sums["pixels"] += error.numel()
        self._test_sums["frames"] += batch_y.shape[0] * batch_y.shape[1]
        self._test_sums["psnr"] += per_frame_psnr.sum()
        self._test_sums["ssim"] += frame_ssim.sum()
        self._test_sums["horizon_squared"] += error.square().sum(dim=(0, 2, 3, 4))
        self._test_sums["horizon_absolute"] += error.abs().sum(dim=(0, 2, 3, 4))
        self._test_sums["horizon_psnr"] += per_frame_psnr.sum(dim=0)
        self._test_sums["horizon_ssim"] += frame_ssim.sum(dim=0)
        self._test_sums["samples"] += batch_y.shape[0]
        if batch_idx == 0:
            self.test_panel = {
                "inputs": batch_x[:4].detach().cpu(),
                "targets": batch_y[:4].detach().cpu(),
                "predictions": prediction[:4].detach().cpu(),
            }

    def on_test_epoch_end(self):
        scalar_names = ("squared", "absolute", "pixels", "frames", "psnr", "ssim", "samples")
        vector_names = (
            "horizon_squared", "horizon_absolute", "horizon_psnr", "horizon_ssim"
        )
        values = torch.cat((
            torch.stack([self._test_sums[name] for name in scalar_names]),
            *[self._test_sums[name] for name in vector_names],
        ))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
        scalar_count = len(scalar_names)
        scalars = dict(zip(scalar_names, [float(value) for value in values[:scalar_count]]))
        horizons = int(self.hparams.aft_seq_length)
        vectors = {}
        offset = scalar_count
        for name in vector_names:
            vectors[name] = values[offset:offset + horizons].cpu().tolist()
            offset += horizons
        squared, absolute = scalars["squared"], scalars["absolute"]
        pixels, frames, samples = scalars["pixels"], scalars["frames"], scalars["samples"]
        pixels_per_frame = pixels / frames
        metrics = {
            "mse": squared / frames,
            "mae": absolute / frames,
            "pixel_mse": squared / pixels,
            "pixel_mae": absolute / pixels,
            "rmse": (squared / frames) ** 0.5,
            "psnr": scalars["psnr"] / frames,
            "ssim": scalars["ssim"] / frames,
            "horizon_mse": [value / samples for value in vectors["horizon_squared"]],
            "horizon_mae": [value / samples for value in vectors["horizon_absolute"]],
            "horizon_pixel_mse": [
                value / (samples * pixels_per_frame) for value in vectors["horizon_squared"]
            ],
            "horizon_pixel_mae": [
                value / (samples * pixels_per_frame) for value in vectors["horizon_absolute"]
            ],
            "horizon_psnr": [value / samples for value in vectors["horizon_psnr"]],
            "horizon_ssim": [value / samples for value in vectors["horizon_ssim"]],
            "samples_times_frames": int(frames),
        }
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and name != "samples_times_frames":
                self.log(f"test_{name}", value, sync_dist=False)
        if self.trainer.is_global_zero:
            folder = osp.join(self.hparams.save_dir, "saved")
            os.makedirs(folder, exist_ok=True)
            path = osp.join(folder, "streaming_metrics.json")
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
                handle.write("\n")
            os.replace(temporary, path)
        self.streaming_test_metrics = metrics
        return metrics


__all__ = ["HVPIntegrated"]
