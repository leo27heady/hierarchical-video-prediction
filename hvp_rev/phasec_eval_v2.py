"""Protected long Moving-MNIST rollout with persistent-object, intention-to-score endpoints."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from hvp_rev import hier_convae, rhvd, viz, wb
import hvp_rev.step1_mech as mechanism
from hvp_rev.digit_classifier import DigitCNN
from hvp_rev.evaluation import evaluate_frames, score_objects
from hvp_rev.phasec_v2 import classify_components, extract_components, metadata_objects
from hvp_rev.provenance import (
    ArtifactStore,
    RunSpec,
    assert_dataset_purpose,
    atomic_save_npz,
    atomic_write_json,
    log_holdout_access,
    sha256_file,
    sha256_value,
)
from hvp_rev.step1_mech import ActionHierData, decode01


FRAME_KEYS = (
    "fg_nonempty", "pred_fg_nonempty", "mse_full", "mse_gt_fg_nonempty", "mse_gt_bg",
    "mse_union", "fg_laplacian_energy_nonempty", "fg_precision", "fg_recall", "fg_f1",
    "gt_component_count", "pred_component_count", "missed_components", "extra_components",
    "matched_component_iou", "matched_centroid_l2",
)
OBJECT_KEYS = (
    "n_gt", "n_pred", "n_matched", "missed", "extra", "count_error", "coverage",
    "matched_trajectory_error", "trajectory_error", "identity_scorable", "identity_coverage",
    "identity_accuracy_matched", "identity_accuracy",
)


def _resolve_spec(args: argparse.Namespace, manifest: dict) -> RunSpec:
    cfg = rhvd.arm_cfg(args.arm)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 2:
        raise ValueError("E6 requires an RNG-complete provenance-v2 predictor checkpoint")
    trained = checkpoint["run_spec"]
    if trained["arm"] != args.arm or int(trained["seed"]) != args.seed:
        raise ValueError("predictor checkpoint arm/seed differs from evaluation cell")
    if trained["ae_hash"] != sha256_file(args.ae):
        raise ValueError("predictor checkpoint was trained with a different codec")
    artifacts = manifest["artifacts"]
    return RunSpec(
        study="phase-c-v2-evaluation",
        arm=args.arm,
        seed=args.seed,
        dataset_hash=sha256_value({
            "clips": artifacts["clips"]["sha256"],
            "actions": artifacts["actions"]["sha256"],
            "metadata": artifacts["metadata"]["sha256"],
        }),
        ae_hash=sha256_file(args.ae),
        arm_config={
            "legacy_tuple": list(cfg),
            "checkpoint": Path(args.checkpoint).as_posix(),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "classifier_sha256": sha256_file(args.classifier),
            "dataset_id": manifest["dataset_id"],
            "detection_threshold": args.detection_threshold,
            "min_component_pixels": args.min_component_pixels,
            "confidence_threshold": args.confidence_threshold,
            "assignment": {
                "distance_weight": args.distance_weight,
                "iou_weight": args.iou_weight,
                "max_cost": args.max_assignment_cost,
            },
            "evaluation_clip_start": args.eval_start,
        },
        train_examples=int(trained["train_examples"]),
        optimizer_updates=int(trained["optimizer_updates"]),
        batch_size=int(trained["batch_size"]),
        precision=str(trained["precision"]),
        nfe=args.nfe,
        warmup_frames=args.warm,
        rollout_frames=int(manifest["configuration"]["frames"] - args.warm),
        evaluation_clips=args.n_eval_clips,
        source_commit=args.source_commit,
        metric_version="phasec-persistent-object-v2",
        optimizer=trained["optimizer"],
        ema=trained["ema"],
        sampler={
            "name": "DDIM",
            "nfe": args.nfe,
            "sampler_seed": args.sampler_seed,
            "crn": "100000 + 1000000*sampler_seed + 1000*chunk + rollout_step",
        },
        tag="phase-c-v2-protected-boundary",
    )


@torch.no_grad()
def _rollout(args, store: ArtifactStore, train, test, ae, classifier, metadata) -> Path:
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f, n_ctx_t, stride_t = rhvd.arm_cfg(args.arm)
    if mode != "diff" or n_ctx_t:
        raise ValueError("E6 registers only the two two-level diffusion arms")
    net = rhvd.WindowDiT(
        n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c,
        n_hist_f=n_hist_f, n_ctx_t=n_ctx_t, stride_t=stride_t,
    ).to(rhvd.DEV)
    state = torch.load(args.checkpoint, map_location=rhvd.DEV, weights_only=False)
    net.load_state_dict(state["ema"])
    net.eval()
    wrapper = rhvd.WindowDiffusion(net)

    n_frames = rhvd.ctx_reach(n_ctx_c, n_ctx_f, stride_c, n_hist_f, n_ctx_t, stride_t)
    if args.warm < n_frames or (args.warm % stride_c):
        raise ValueError("warm-up does not cover/phase-align the registered context")
    if (test.T - args.warm) % rhvd.WIN:
        raise ValueError("rollout horizon must be divisible by the model window")
    clips = np.arange(args.eval_start, args.eval_start + args.n_eval_clips, dtype=np.int64)
    if clips[-1] >= test.N:
        raise ValueError("registered evaluation clips exceed the protected dataset")
    steps = (test.T - args.warm) // rhvd.WIN
    records: dict[str, list] = {
        "clip": [], "h": [], "frame": [], "overlap_any": [],
        **{key: [] for key in FRAME_KEYS},
        **{f"object_{key}": [] for key in OBJECT_KEYS},
    }
    started = time.time()
    visual_rows = []
    for c0 in range(0, len(clips), args.chunk):
        elapsed = time.time() - started
        if args.authorized_gpu_hours > 0 and c0:
            chunks_done = c0 // args.chunk
            if (elapsed + elapsed / chunks_done) / 3600 > args.authorized_gpu_hours:
                raise TimeoutError("next E6 rollout chunk is projected beyond its cell authorization")
        selected = clips[c0:c0 + args.chunk]
        batch = len(selected)
        if stride_c > 1:
            buf_c = test.zc[
                selected,
                args.warm - n_ctx_c * stride_c + stride_c - 1:args.warm:stride_c,
            ].clone().to(rhvd.DEV)
        else:
            buf_c = test.zc[selected, args.warm - n_frames:args.warm].clone().to(rhvd.DEV)
        buf_f = test.zf[selected, args.warm - n_frames:args.warm].clone().to(rhvd.DEV)

        for rollout_step in range(steps):
            first_frame = args.warm + rollout_step * rhvd.WIN
            torch.manual_seed(
                rhvd.rollout_crn_seed(args.sampler_seed, c0 // args.chunk, rollout_step)
            )
            actions = torch.from_numpy(
                test.acts[selected, first_frame:first_frame + rhvd.WIN].astype(np.int64)
            ).to(rhvd.DEV)
            output_c, output_f = wrapper.sample(
                buf_c[:, -n_ctx_c:] if n_ctx_c else buf_c[:, :0],
                buf_f[:, -n_ctx_f:],
                actions,
                n_steps=args.nfe,
            )
            keep_c = n_ctx_c if stride_c > 1 else n_frames
            if n_ctx_c:
                buf_c = torch.cat([buf_c, output_c], 1)[:, -keep_c:]
            buf_f = torch.cat([buf_f, output_f], 1)[:, -n_frames:]

            for offset in range(rhvd.WIN):
                frame_index = first_frame + offset
                pred = decode01(ae, output_f[:, offset]).detach().cpu().numpy()[:, 0]
                truth = test.clips_u8[selected, frame_index].astype(np.float32) / 255.0
                frame_values = evaluate_frames(
                    pred[:, None], truth[:, None],
                    threshold=args.detection_threshold,
                    min_component_pixels=args.min_component_pixels,
                )
                components = [
                    extract_components(
                        item,
                        threshold=args.detection_threshold,
                        min_component_pixels=args.min_component_pixels,
                    )
                    for item in pred
                ]
                lengths = [len(item) for item in components]
                flat = [component for item in components for component in item]
                classified = classify_components(
                    classifier, flat, device=rhvd.DEV,
                    confidence_threshold=args.confidence_threshold,
                )
                cursor = 0
                for local, clip in enumerate(selected):
                    detections = classified[cursor:cursor + lengths[local]]
                    cursor += lengths[local]
                    gt = metadata_objects(
                        object_id=metadata["object_id"],
                        class_id=metadata["class_id"],
                        bbox_yxyx=metadata["bbox_yxyx"],
                        clip=int(clip),
                        frame=frame_index,
                    )
                    object_values = score_objects(
                        detections, gt, frame_shape=(64, 64),
                        distance_weight=args.distance_weight,
                        iou_weight=args.iou_weight,
                        max_assignment_cost=args.max_assignment_cost,
                    )
                    records["clip"].append(int(clip))
                    records["h"].append(frame_index - args.warm + 1)
                    records["frame"].append(frame_index)
                    records["overlap_any"].append(
                        bool(metadata["overlap"][clip, frame_index].any())
                    )
                    for key in FRAME_KEYS:
                        records[key].append(np.asarray(frame_values[key]).reshape(-1)[local])
                    for key in OBJECT_KEYS:
                        records[f"object_{key}"].append(object_values[key])
                if c0 == 0 and (frame_index - args.warm + 1) in {1, 48, 150, 300}:
                    visual_rows.append(
                        (frame_index - args.warm + 1,
                         torch.from_numpy(pred[:6])[:, None],
                         torch.from_numpy(truth[:6])[:, None])
                    )
        elapsed = time.time() - started
        fg = np.asarray(records["mse_gt_fg_nonempty"], np.float64)
        wb.log({
            "health/clips_complete": int(c0 + batch),
            "health/clips_total": len(clips),
            "health/progress": float((c0 + batch) / len(clips)),
            "health/elapsed_hours": elapsed / 3600,
            "health/running_fg_mse": float(np.nanmean(fg)),
            "health/running_object_coverage": float(np.mean(records["object_coverage"])),
        }, step=int(c0 + batch))
        print(
            f"[phasec-v2 {args.arm} s{args.seed}] clips {c0}..{c0 + batch - 1}; "
            f"{elapsed/3600:.2f}h; fg={np.nanmean(fg):.4f}; "
            f"coverage={np.mean(records['object_coverage']):.4f}",
            flush=True,
        )
        if args.authorized_gpu_hours > 0 and elapsed / 3600 > args.authorized_gpu_hours:
            raise TimeoutError("E6 rollout exceeded its cell authorization")

    arrays = {key: np.asarray(value) for key, value in records.items()}
    output = store.path / "metrics" / "phasec_v2_records.npz"
    atomic_save_npz(output, **arrays)
    if visual_rows:
        panels = []
        for horizon, pred, truth in visual_rows:
            panels.extend([(f"pred_h{horizon}", pred), (f"GT_h{horizon}", truth)])
        visual = store.path / "diagnostics" / "rollout_strip.png"
        visual.parent.mkdir(parents=True, exist_ok=True)
        viz.save_samples(str(visual), panels, n=6)
        wb.image("eval/rollout_strip", str(visual))
    wb.summary({
        "eval/fg_mse": float(np.nanmean(arrays["mse_gt_fg_nonempty"])),
        "eval/full_mse": float(np.mean(arrays["mse_full"])),
        "eval/union_mse": float(np.mean(arrays["mse_union"])),
        "eval/object_coverage": float(np.mean(arrays["object_coverage"])),
        "eval/identity_accuracy": float(np.nanmean(arrays["object_identity_accuracy"])),
        "eval/trajectory_error": float(np.mean(arrays["object_trajectory_error"])),
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["mmL-flat", "mmL-hier"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--artifact-root", default="artifacts/runs")
    parser.add_argument("--access-log", default="artifacts/phasec_v2/access.jsonl")
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--warm", type=int, default=48)
    parser.add_argument("--eval-start", type=int, default=250)
    parser.add_argument("--n-eval-clips", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--detection-threshold", type=float, default=0.3)
    parser.add_argument("--min-component-pixels", type=int, default=30)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--distance-weight", type=float, default=0.5)
    parser.add_argument("--iou-weight", type=float, default=0.5)
    parser.add_argument("--max-assignment-cost", type=float, default=0.65)
    parser.add_argument("--authorized-gpu-hours", type=float, required=True)
    args = parser.parse_args()

    if rhvd.DEV != "cuda":
        raise RuntimeError("protected E6 evaluation requires CUDA")
    mechanism._BF16 = True
    manifest = assert_dataset_purpose(
        args.holdout_manifest,
        allowed={"phase_c_v2_holdout"},
        operation="Phase C v2 protected evaluation",
    )
    spec = _resolve_spec(args, manifest)
    store = ArtifactStore(args.artifact_root, spec)
    log_holdout_access(
        args.holdout_manifest, args.access_log,
        run_spec_hash=spec.identity,
        allowed_purpose="phase_c_v2_holdout",
        operation="Phase C v2 protected evaluation",
    )
    for name in ("clips", "actions", "metadata"):
        path = manifest["artifacts"][name]["path"]
        if sha256_file(path) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"protected {name} hash differs from its manifest")

    run_id = f"phasec-v2-eval-{spec.identity}"
    wb.init(
        f"phasec-v2-{args.arm}-s{args.seed}",
        config={"run_spec": spec.resolved(), "cli": vars(args)},
        project=args.wandb_project,
        run_id=run_id,
        group=args.wandb_group,
        tags=["moving-mnist", "phase-c-v2", "persistent-object", "protected-eval"],
        job_type="phasec-v2-evaluation",
        required=True,
        resume="never",
    )
    active = Path("artifacts/active_experiment.json")
    atomic_write_json(active, {
        "schema_version": 1, "status": "running", "study": spec.study,
        "arm": args.arm, "seed": args.seed, "optimizer_updates": spec.optimizer_updates,
        "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec_hash": spec.identity, "run_dir": store.path.as_posix(),
        "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
    })
    try:
        ae = hier_convae.load(args.ae)
        train = ActionHierData(
            ae, args.train_path, args.train_path.replace(".npy", "_actions.npy"),
            n_ctx=6, k=4, n_val=250,
        )
        test_path = manifest["artifacts"]["clips"]["path"]
        test = ActionHierData(
            ae, test_path, manifest["artifacts"]["actions"]["path"],
            n_ctx=6, k=4, n_val=250, stats=train.stats,
        )
        classifier = DigitCNN().to(rhvd.DEV)
        classifier.load_state_dict(torch.load(args.classifier, map_location=rhvd.DEV, weights_only=True))
        classifier.eval()
        with np.load(manifest["artifacts"]["metadata"]["path"]) as source:
            metadata = {key: source[key] for key in source.files}
        result = _rollout(args, store, train, test, ae, classifier, metadata)
        store.write_json("evaluation.json", {
            "result_path": result.as_posix(),
            "dataset_manifest": Path(args.holdout_manifest).resolve().as_posix(),
            "dataset_manifest_sha256": sha256_file(args.holdout_manifest),
            "access_log": Path(args.access_log).resolve().as_posix(),
            "wandb_url": wb.active_url(),
        })
        store.finalize()
        wb.artifact(
            f"phasec-v2-eval-{spec.identity}",
            "phasec-v2-evaluation",
            [
                str(store.path / "run_spec.json"), str(result),
                str(store.path / "evaluation.json"), str(store.path / "manifest.json"),
                args.holdout_manifest,
            ],
            metadata={"run_spec_hash": spec.identity, "arm": args.arm, "seed": args.seed},
            aliases=["latest", f"{args.arm}-s{args.seed}"],
        )
        atomic_write_json(active, {
            "schema_version": 1, "status": "complete", "study": spec.study,
            "arm": args.arm, "seed": args.seed, "optimizer_updates": spec.optimizer_updates,
            "pid": os.getpid(), "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_spec_hash": spec.identity, "run_dir": store.path.as_posix(),
            "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
        })
        print(f"PHASEC_V2_EVAL_DONE {args.arm} s{args.seed} {spec.identity}", flush=True)
    except Exception:
        atomic_write_json(active, {
            "schema_version": 1, "status": "failed", "study": spec.study,
            "arm": args.arm, "seed": args.seed, "optimizer_updates": spec.optimizer_updates,
            "pid": os.getpid(), "failed_utc": datetime.now(timezone.utc).isoformat(),
            "run_spec_hash": spec.identity, "run_dir": store.path.as_posix(),
            "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
        })
        raise
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
