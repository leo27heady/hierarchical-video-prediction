"""Reproduce a same-population all-black diagnostic for the archived CW-VAE curve.

The archived CW-VAE evaluator selected the best of several stochastic samples by
clip-average SSIM, then retained only the mean SSIM/PSNR curve.  Consequently this
script can reproduce the deterministic baseline on the exact clips, frames and
metric implementation, but it cannot form paired confidence intervals against
CW-VAE because the model's record-level scores were not retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def black_frame_metrics(gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-record SSIM/PSNR against zeros using the CW-VAE metric semantics."""
    if gt.ndim != 4:
        raise ValueError(f"expected (clips, frames, height, width), got {gt.shape}")
    if gt.dtype != np.float32 and gt.dtype != np.float64:
        raise TypeError("ground truth must be normalized floating point")
    clips, frames = gt.shape[:2]
    black = np.zeros(gt.shape[2:], dtype=gt.dtype)
    ssim = np.empty((clips, frames), dtype=np.float64)
    for clip in range(clips):
        for frame in range(frames):
            ssim[clip, frame] = structural_similarity(
                gt[clip, frame], black, data_range=1.0
            )
    mse = np.mean(np.square(gt, dtype=np.float64), axis=(2, 3))
    with np.errstate(divide="ignore"):
        psnr = np.where(mse == 0.0, 100.0, 20.0 * np.log10(1.0 / np.sqrt(mse)))
    return ssim, psnr


def summarize_window(values: np.ndarray, start: int, stop: int) -> dict[str, float | int]:
    selected = values[:, start:stop]
    return {
        "start_frame_inclusive": start + 1,
        "stop_frame_inclusive": stop,
        "records": int(selected.size),
        "mean": float(selected.mean()),
        "clip_mean_sd": float(selected.mean(axis=1).std(ddof=1)),
    }


def model_window(curve: np.ndarray, start: int, stop: int) -> dict[str, float | int]:
    selected = curve[start:stop]
    return {
        "start_frame_inclusive": start + 1,
        "stop_frame_inclusive": stop,
        "records_represented": int(selected.size),
        "mean": float(selected.mean()),
    }


def markdown(result: dict) -> str:
    lines = [
        "# CW-VAE same-population all-black diagnostic",
        "",
        (
            f"The archived PyTorch CW-VAE run used {result['protocol']['clips']} clips, "
            f"{result['protocol']['context']} context frames, a "
            f"{result['protocol']['horizon']}-frame open-loop horizon and best-of-"
            f"{result['protocol']['cwvae_samples']} selection by clip-average SSIM."
        ),
        "",
        "| Horizon window | CW-VAE SSIM | Black SSIM | CW-VAE PSNR | Black PSNR |",
        "|---|---:|---:|---:|---:|",
    ]
    for window_id, black_ssim in result["black"]["ssim"]["windows"].items():
        cw_ssim = result["cwvae"]["ssim"]["windows"][window_id]
        black_psnr = result["black"]["psnr"]["windows"][window_id]
        cw_psnr = result["cwvae"]["psnr"]["windows"][window_id]
        label = (
            f"+{black_ssim['start_frame_inclusive']}–"
            f"{black_ssim['stop_frame_inclusive']}"
        )
        lines.append(
            f"| {label} | {cw_ssim['mean']:.4f} | {black_ssim['mean']:.4f} | "
            f"{cw_psnr['mean']:.3f} | {black_psnr['mean']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        (
            "This is a metric-sanity diagnostic, not a model-ranking experiment. It uses the "
            "same clips, target frames and `skimage` SSIM semantics as the archived evaluator, "
            "but the CW-VAE score is best-of-stochastic-samples whereas black is deterministic. "
            "Architectures, training budgets and inference budgets are not matched."
        ),
        "",
        (
            "Only the CW-VAE mean curves survived, so paired record- or clip-level confidence "
            "intervals cannot be reconstructed. High black-frame scores on sparse Moving-MNIST "
            "mainly expose background dominance in full-frame SSIM/PSNR; they do not imply useful "
            "motion or object prediction."
        ),
        "",
        "## Provenance",
        "",
        f"- dataset SHA-256: `{result['inputs']['dataset']['sha256']}`",
        f"- archived curve SHA-256: `{result['inputs']['cwvae_curve']['sha256']}`",
        f"- archived evaluator SHA-256: `{result['inputs']['cwvae_evaluator']['sha256']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../cwvae/data_mmnist/test.npy")
    parser.add_argument(
        "--curve", default="../cwvae/logs_torch/cwvae/eval/openloop_curve.json"
    )
    parser.add_argument("--evaluator", default="../cwvae/eval_torch.py")
    parser.add_argument(
        "--out-json", default="artifacts/cwvae_black_baseline/results.json"
    )
    parser.add_argument(
        "--out-md", default="docs/confirmation/CWVAE_BLACK_BASELINE.md"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    curve_path = Path(args.curve)
    evaluator_path = Path(args.evaluator)
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    clips = int(curve["clips"])
    context = int(curve["ctx"])
    horizon = int(curve["horizon"])
    cw_ssim = np.asarray(curve["ssim"], dtype=np.float64)
    cw_psnr = np.asarray(curve["psnr"], dtype=np.float64)
    if cw_ssim.shape != (horizon,) or cw_psnr.shape != (horizon,):
        raise ValueError("archived CW-VAE curves do not match their declared horizon")

    data = np.load(dataset_path, mmap_mode="r")
    if data.ndim != 4 or data.shape[0] < clips or data.shape[1] < context + horizon:
        raise ValueError(
            f"dataset {data.shape} cannot supply {clips}x(context {context}+horizon {horizon})"
        )
    if data.dtype != np.uint8:
        raise TypeError(f"expected uint8 Moving-MNIST data, got {data.dtype}")
    gt = np.asarray(
        data[:clips, context : context + horizon], dtype=np.float32
    ) / 255.0
    black_ssim, black_psnr = black_frame_metrics(gt)

    raw_windows = [(0, min(48, horizon)), (48, min(192, horizon)), (192, horizon)]
    windows = [(start, stop) for start, stop in raw_windows if stop > start]
    window_ids = [f"h{start + 1}_{stop}" for start, stop in windows]

    result = {
        "schema_version": 1,
        "status": "SAME_POPULATION_DIAGNOSTIC_COMPLETE",
        "scope": (
            "Metric-sanity diagnostic only; not architecture-, training- or inference-budget "
            "matched and no paired CW-VAE inference is possible from mean curves."
        ),
        "protocol": {
            "clips": clips,
            "context": context,
            "horizon": horizon,
            "cwvae_samples": int(curve["n_samples"]),
            "cwvae_selection": "best clip-average SSIM among stochastic samples",
            "black_prediction": "all pixels zero for every predicted frame",
            "ssim": "skimage.metrics.structural_similarity(data_range=1.0)",
            "psnr": "20*log10(1/sqrt(full-frame MSE)); 100 when MSE is zero",
        },
        "black": {
            "ssim": {
                "overall_mean": float(black_ssim.mean()),
                "clip_mean_sd": float(black_ssim.mean(axis=1).std(ddof=1)),
                "windows": {
                    key: summarize_window(black_ssim, start, stop)
                    for key, (start, stop) in zip(window_ids, windows)
                },
            },
            "psnr": {
                "overall_mean": float(black_psnr.mean()),
                "clip_mean_sd": float(black_psnr.mean(axis=1).std(ddof=1)),
                "windows": {
                    key: summarize_window(black_psnr, start, stop)
                    for key, (start, stop) in zip(window_ids, windows)
                },
            },
        },
        "cwvae": {
            "ssim": {
                "overall_mean": float(cw_ssim.mean()),
                "windows": {
                    key: model_window(cw_ssim, start, stop)
                    for key, (start, stop) in zip(window_ids, windows)
                },
            },
            "psnr": {
                "overall_mean": float(cw_psnr.mean()),
                "windows": {
                    key: model_window(cw_psnr, start, stop)
                    for key, (start, stop) in zip(window_ids, windows)
                },
            },
        },
        "diagnostics": {
            "blank_ground_truth_records": int(
                np.count_nonzero(np.mean(np.square(gt, dtype=np.float64), axis=(2, 3)) == 0)
            ),
            "finite": bool(
                np.isfinite(black_ssim).all()
                and np.isfinite(black_psnr).all()
                and np.isfinite(cw_ssim).all()
                and np.isfinite(cw_psnr).all()
            ),
        },
        "inputs": {
            "dataset": {
                "path": dataset_path.as_posix(),
                "sha256": sha256_file(dataset_path),
                "shape": list(data.shape),
                "dtype": str(data.dtype),
            },
            "cwvae_curve": {
                "path": curve_path.as_posix(),
                "sha256": sha256_file(curve_path),
            },
            "cwvae_evaluator": {
                "path": evaluator_path.as_posix(),
                "sha256": sha256_file(evaluator_path),
            },
        },
    }
    if not result["diagnostics"]["finite"]:
        raise ValueError("non-finite CW-VAE/black diagnostic")
    if not math.isclose(
        result["cwvae"]["ssim"]["overall_mean"],
        0.5707673059662415,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("archived CW-VAE curve differs from the audited source")

    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "black_ssim": result["black"]["ssim"]["overall_mean"],
        "cwvae_ssim": result["cwvae"]["ssim"]["overall_mean"],
        "black_psnr": result["black"]["psnr"]["overall_mean"],
        "cwvae_psnr": result["cwvae"]["psnr"]["overall_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
