"""Evaluate the frozen E7a codec validity floor before predictor training or holdout access."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import hier_convae, wb
from hvp_rev.provenance import atomic_write_json, sha256_file


@torch.no_grad()
def evaluate(path: str, frames: np.ndarray, batch_size: int, threshold: float) -> dict:
    ae = hier_convae.load(path)
    sums = {
        "fine_full_sse": 0.0, "coarse_full_sse": 0.0,
        "fine_fg_sse": 0.0, "coarse_fg_sse": 0.0,
        "full_count": 0, "fg_count": 0,
    }
    z_sum = {"coarse": torch.zeros(hier_convae.LAT, dtype=torch.float64),
             "fine": torch.zeros(hier_convae.LAT, dtype=torch.float64)}
    z_sq = {"coarse": torch.zeros(hier_convae.LAT, dtype=torch.float64),
            "fine": torch.zeros(hier_convae.LAT, dtype=torch.float64)}
    z_count = {"coarse": 0, "fine": 0}
    finite = True
    for start in range(0, len(frames), batch_size):
        raw = np.asarray(frames[start:start + batch_size], dtype=np.float32) / 255.0
        gt01 = torch.from_numpy(raw).unsqueeze(1).to(hier_convae.DEV)
        x = gt01 * 2 - 1
        rc, rf, zc, zf = ae(x)
        rc01, rf01 = (rc + 1) * 0.5, (rf + 1) * 0.5
        ec, ef = (rc01 - gt01).square(), (rf01 - gt01).square()
        fg = gt01 > threshold
        sums["coarse_full_sse"] += float(ec.sum())
        sums["fine_full_sse"] += float(ef.sum())
        sums["full_count"] += int(ec.numel())
        sums["coarse_fg_sse"] += float(ec[fg].sum())
        sums["fine_fg_sse"] += float(ef[fg].sum())
        sums["fg_count"] += int(fg.sum())
        for name, latent in (("coarse", zc), ("fine", zf)):
            finite = finite and bool(torch.isfinite(latent).all())
            values = latent.detach().double().permute(1, 0, 2, 3).reshape(hier_convae.LAT, -1).cpu()
            z_sum[name] += values.sum(dim=1)
            z_sq[name] += values.square().sum(dim=1)
            z_count[name] += values.shape[1]
    result = {
        "fine_full_mse": sums["fine_full_sse"] / sums["full_count"],
        "coarse_full_mse": sums["coarse_full_sse"] / sums["full_count"],
        "fine_foreground_mse": sums["fine_fg_sse"] / sums["fg_count"],
        "coarse_foreground_mse": sums["coarse_fg_sse"] / sums["fg_count"],
        "foreground_pixels": sums["fg_count"],
        "finite": finite,
    }
    for name in ("coarse", "fine"):
        mean = z_sum[name] / z_count[name]
        variance = (z_sq[name] / z_count[name] - mean.square()).clamp_min(0)
        std = variance.sqrt().numpy()
        result[f"{name}_channel_std"] = std.tolist()
        result[f"{name}_minimum_channel_std"] = float(std.min())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--alternate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    floor = config["codec_floor"]
    alt_meta = torch.load(args.alternate, map_location="cpu", weights_only=False).get("training", {})
    expected_alt = config["codecs"]["alternate"]
    if any(alt_meta.get(key) != expected_alt[key] for key in ("seed", "steps", "batch", "lr")):
        raise ValueError(f"alternate codec training metadata differs from frozen design: {alt_meta}")
    if sha256_file(args.original) != config["codecs"]["original"]["sha256"]:
        raise ValueError("original codec hash changed")
    data = np.load(expected_alt["training_data"], mmap_mode="r")
    frames = data.reshape(-1, 64, 64)[-4000:]
    original = evaluate(
        args.original, frames, int(floor["batch_size"]), float(floor["metric_threshold"])
    )
    alternate = evaluate(
        args.alternate, frames, int(floor["batch_size"]), float(floor["metric_threshold"])
    )
    relative = float(floor["relative_multiplier"])
    limits = {
        "fine_full_mse": max(relative * original["fine_full_mse"],
                             original["fine_full_mse"] + floor["fine_full_absolute_allowance"]),
        "coarse_full_mse": max(relative * original["coarse_full_mse"],
                               original["coarse_full_mse"] + floor["coarse_full_absolute_allowance"]),
        "fine_foreground_mse": max(
            relative * original["fine_foreground_mse"],
            original["fine_foreground_mse"] + floor["fine_foreground_absolute_allowance"],
        ),
        "coarse_foreground_mse": max(
            relative * original["coarse_foreground_mse"],
            original["coarse_foreground_mse"] + floor["coarse_foreground_absolute_allowance"],
        ),
    }
    failures = [
        f"{metric} {alternate[metric]:.8f} exceeds {limit:.8f}"
        for metric, limit in limits.items() if alternate[metric] > limit
    ]
    if not original["finite"] or not alternate["finite"]:
        failures.append("non-finite codec latent")
    minimum_std = float(floor["minimum_channel_std"])
    for codec_name, values in (("original", original), ("alternate", alternate)):
        for level in ("coarse", "fine"):
            value = values[f"{level}_minimum_channel_std"]
            if value < minimum_std:
                failures.append(f"{codec_name} {level} latent channel std {value:.6f} < {minimum_std}")
    report = {
        "schema_version": 1,
        "status": "CODEC_FLOOR_PASS" if not failures else "CODEC_FLOOR_FAIL",
        "population": floor["population"],
        "frames": 4000,
        "config_sha256": sha256_file(args.config),
        "original_codec_sha256": sha256_file(args.original),
        "alternate_codec_sha256": sha256_file(args.alternate),
        "original": original,
        "alternate": alternate,
        "limits": limits,
        "failures": failures,
    }
    atomic_write_json(args.out, report)
    run_id = f"e7a-codec-floor-{report['alternate_codec_sha256'][:12]}"
    wb.init(
        run_id, project=args.wandb_project, run_id=run_id, resume="never",
        group=args.wandb_group, job_type="codec-validity-gate", required=True,
        tags=["loopmom", "alternate-codec-robustness", "codec-floor"],
        config={"config_sha256": report["config_sha256"], "limits": limits},
    )
    wb.summary({
        "floor/status": report["status"],
        **{f"original/{key}": value for key, value in original.items()
           if isinstance(value, (int, float, bool))},
        **{f"alternate/{key}": value for key, value in alternate.items()
           if isinstance(value, (int, float, bool))},
    })
    wb.artifact(
        run_id, "codec-validity-report", [args.out, args.config],
        metadata={"status": report["status"]}, aliases=["latest"],
    )
    wb.finish()
    print(json.dumps({key: report[key] for key in ("status", "limits", "failures")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
