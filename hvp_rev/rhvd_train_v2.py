"""Provenance-v2 controller around the numerically unchanged RHVD training kernel."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from hvp_rev import arm_registry, hier_convae, rhvd, viz, wb
import hvp_rev.step1_mech as mechanism
from hvp_rev.provenance import (
    ArtifactStore,
    RunSpec,
    atomic_write_json,
    load_training_checkpoint,
    save_training_checkpoint,
    sha256_file,
    sha256_value,
)
from hvp_rev.step1_mech import ActionHierData, decode01, fg_mse_px_frames


def _data_artifacts(train_path: str, test_path: str) -> dict[str, str]:
    """Hash every array that can affect training, rollout records, or geometric labels."""
    train_actions = train_path.replace(".npy", "_actions.npy")
    test_actions = test_path.replace(".npy", "_actions.npy")
    paths = {
        "train_clips": train_path,
        "train_actions": train_actions,
        "test_clips": test_path,
        "test_actions": test_actions,
    }
    for split, clip_path in (("train", train_path), ("test", test_path)):
        positions = clip_path.replace(".npy", "_positions.npy")
        if Path(positions).is_file():
            paths[f"{split}_positions"] = positions
    return {name: sha256_file(path) for name, path in paths.items()}


def resolved_spec(args: argparse.Namespace) -> RunSpec:
    artifacts = _data_artifacts(args.train_path, args.test_path)
    registered = arm_registry.from_legacy_mapping(rhvd.ARMS)[args.arm].resolved()
    registered["data_artifacts"] = artifacts
    test_shape = np.load(args.test_path, mmap_mode="r").shape
    return RunSpec(
        study=args.study,
        arm=args.arm,
        seed=args.seed,
        dataset_hash=sha256_value(artifacts),
        ae_hash=sha256_file(args.ae_path),
        arm_config=registered,
        train_examples=args.steps * args.batch,
        optimizer_updates=args.steps,
        batch_size=args.batch,
        precision="bf16" if args.bf16 else "fp32",
        nfe=args.nfe,
        warmup_frames=args.warm,
        rollout_frames=int(test_shape[1] - args.warm),
        evaluation_clips=args.n_eval_clips,
        source_commit=args.source_commit,
        metric_version="complete-v2-thr0.5",
        optimizer={
            "name": "AdamW", "lr": args.lr, "weight_decay": 1e-4,
            "warmup_updates": args.warmup, "gradient_clip_norm": 1.0,
        },
        ema={"decay": 0.999, "evaluation": "EMA"},
        sampler={
            "training_order": "global NumPy RNG; clip then target-start per sample",
            "rollout": "DDIM" if rhvd.arm_cfg(args.arm)[2] == "diff" else "deterministic-regression",
            "nfe": args.nfe,
            "rollout_crn": "100000 + 1000*chunk_index + rollout_step",
        },
        tag=args.provenance_tag,
    )


def _save(path: Path, spec: RunSpec, step: int, net, opt, ema, data, batch: int) -> None:
    save_training_checkpoint(
        path,
        spec=spec,
        step=step,
        model=net.state_dict(),
        optimizer=opt.state_dict(),
        ema=ema.shadow.state_dict(),
        sampler_state={"examples_seen": int(step * batch), "next_update": int(step + 1)},
        extra={"latent_stats": data.stats},
    )


def train(args: argparse.Namespace, spec: RunSpec, store: ArtifactStore, data, test, ae):
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f, n_ctx_t, stride_t = rhvd.arm_cfg(args.arm)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net = rhvd.WindowDiT(
        n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c,
        n_hist_f=n_hist_f, n_ctx_t=n_ctx_t, stride_t=stride_t,
    ).to(rhvd.DEV)
    wrap = rhvd.WindowDiffusion(net) if mode == "diff" else rhvd.WindowRegression(net)
    ema = rhvd.EMA(net)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    checkpoint = store.path / "checkpoint.pt"
    start = 1
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError(f"partial checkpoint exists; pass --resume explicitly: {checkpoint}")
        state = load_training_checkpoint(checkpoint, spec, restore_rng=True)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        ema.shadow.load_state_dict(state["ema"])
        saved_stats = state.get("extra", {}).get("latent_stats")
        if saved_stats is None or any(
            not torch.equal(saved.cpu(), current.cpu())
            for saved, current in zip(saved_stats, data.stats)
        ):
            raise ValueError("latent normalization statistics differ on resume")
        start = int(state["step"]) + 1
        rhvd.log(f"[v2 resume] {checkpoint} @ {state['step']}")
    elif args.resume:
        raise FileNotFoundError(f"--resume requested but no checkpoint exists: {checkpoint}")

    rhvd.log(
        f"[v2 train] {args.arm} {start}..{args.steps} B={args.batch} seed={args.seed} "
        f"RunSpec={spec.identity} params={sum(p.numel() for p in net.parameters())/1e6:.2f}M"
    )
    if start > args.steps:
        return wrap, ema
    if rhvd.DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    net.train()
    running = 0.0
    last = time.time()
    for step in range(start, args.steps + 1):
        batch = rhvd.window_batch(
            data, args.batch, n_ctx_c, n_ctx_f, train=True, stride_c=stride_c,
            n_hist_f=n_hist_f, n_ctx_t=n_ctx_t, stride_t=stride_t,
        )
        loss = rhvd.training_update(
            wrap, net, ema, opt, batch, step, lr=args.lr, warmup=args.warmup,
        )
        running += float(loss.detach())
        elapsed_step = time.time() - last
        last = time.time()
        if elapsed_step > rhvd.SPILL_SEC and step > start + 3:
            _save(checkpoint, spec, step, net, opt, ema, data, args.batch)
            raise RuntimeError(
                f"[v2 {args.arm}] step {step} took {elapsed_step:.1f}s; checkpoint saved"
            )
        if step % args.val_every == 0:
            vram = torch.cuda.max_memory_allocated() / 2**30 if rhvd.DEV == "cuda" else 0.0
            mean_loss = running / args.val_every
            rhvd.log(
                f"  [v2 {args.arm} {step}/{args.steps}] loss {mean_loss:.4f} "
                f"{elapsed_step:.2f}s/step peak {vram:.2f}GB"
            )
            wb.log({
                "train/loss": mean_loss,
                "train/s_per_step": elapsed_step,
                "train/vram_gb": vram,
                "train/lr": opt.param_groups[0]["lr"],
            }, step=step)
            running = 0.0
        if step % args.grid_every == 0:
            net.eval()
            validation = rhvd.window_batch(
                test, 8, n_ctx_c, n_ctx_f, train=False, seed=777, frames=True,
                stride_c=stride_c, n_hist_f=n_hist_f, n_ctx_t=n_ctx_t, stride_t=stride_t,
            )
            eval_wrap = (
                rhvd.WindowDiffusion(ema.shadow, wrap.Ttr)
                if mode == "diff" else rhvd.WindowRegression(ema.shadow)
            )
            output = eval_wrap.sample(
                validation["ctx_c"], validation["ctx_f"], validation["actions"],
                n_steps=args.nfe, ctx_t=validation.get("ctx_t"),
            )
            prediction = decode01(ae, output[1][:, -1])
            ground_truth = (validation["gt"][:, -1].clamp(-1, 1) + 1) * 0.5
            grid = store.path / "diagnostics" / f"train_step_{step}.png"
            grid.parent.mkdir(parents=True, exist_ok=True)
            viz.save_samples(str(grid), [("pred_h4", prediction), ("GT", ground_truth)], n=8)
            validation_fg = float(fg_mse_px_frames(prediction, ground_truth).mean())
            net.train()
            wb.log({"val/fg_mse_h4": validation_fg}, step=step)
            wb.image("val/sample_grid", str(grid), step=step)
        # Capture RNG after all work associated with this update, including diagnostics.
        if step % args.save_every == 0 or step == args.steps:
            _save(checkpoint, spec, step, net, opt, ema, data, args.batch)
    return wrap, ema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(rhvd.ARMS))
    parser.add_argument("--study", default="e4-head-package-seed2")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--n_eval_clips", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train_path", default="data/loopmom_train_3000x100.npy")
    parser.add_argument("--test_path", default="data/loopmom_test_500x200.npy")
    parser.add_argument("--world", type=int, default=192)
    parser.add_argument("--wspeed", type=int, default=4)
    parser.add_argument("--warm", type=int, default=48)
    parser.add_argument("--n_val", type=int, default=250)
    parser.add_argument("--ae_path", default="logs_rhcp/hier_ae_mom.pt")
    parser.add_argument("--artifact_root", default="artifacts/training")
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--wandb_project", default="hgrid")
    parser.add_argument("--wandb_group", required=True)
    parser.add_argument("--dataset_tag", default="loopmom")
    parser.add_argument("--provenance_tag", default="h3b-provenance-v2")
    parser.add_argument("--run_log", default="")
    parser.add_argument("--authorized_gpu_hours", type=float, required=True)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--grid_every", type=int, default=2000)
    args = parser.parse_args()

    mechanism._BF16 = bool(args.bf16)
    if rhvd.DEV != "cuda":
        raise RuntimeError("scientific v2 training requires CUDA")
    spec = resolved_spec(args)
    store = ArtifactStore(args.artifact_root, spec)
    checkpoint = store.path / "checkpoint.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"partial run requires explicit --resume: {store.path}")
    if args.resume and not checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint missing: {checkpoint}")

    wb.init(
        f"{args.study}-{args.arm}-s{args.seed}",
        config={"run_spec": spec.resolved(), "cli": vars(args)},
        project=args.wandb_project,
        run_id=f"{args.study}-{spec.identity}",
        tags=[args.dataset_tag, "training-v2", args.provenance_tag, rhvd.arm_cfg(args.arm)[2]],
        resume="allow" if args.resume else "never",
        group=args.wandb_group,
        job_type="scientific-training",
        required=True,
    )
    active = Path("artifacts/active_experiment.json")
    atomic_write_json(active, {
        "schema_version": 1, "status": "running", "study": args.study,
        "arm": args.arm, "seed": args.seed, "optimizer_updates": args.steps,
        "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec_hash": spec.identity, "run_dir": store.path.as_posix(),
        "run_log": args.run_log, "wandb_url": wb.active_url(),
        "authorized_gpu_hours": args.authorized_gpu_hours,
    })
    try:
        ae = hier_convae.load(args.ae_path)
        data = ActionHierData(
            ae, args.train_path, args.train_path.replace(".npy", "_actions.npy"),
            n_ctx=6, k=4, W=args.world, speed=args.wspeed,
        )
        test = ActionHierData(
            ae, args.test_path, args.test_path.replace(".npy", "_actions.npy"),
            n_ctx=6, k=4, n_val=args.n_val, W=args.world, speed=args.wspeed, stats=data.stats,
        )
        wrap, ema = train(args, spec, store, data, test, ae)
        result = store.path / "metrics" / "rollout_complete.npz"
        rhvd.rollout_eval(
            args.arm, wrap, ema, test, ae, args.seed,
            n_clips=args.n_eval_clips, nfe=args.nfe, warm=args.warm,
            tag=f"_v2-{spec.identity}", complete_metrics=True, output_path=result,
        )
        store.write_json("evaluation.json", {
            "result_path": str(result), "wandb_url": wb.active_url(),
            "metric_version": spec.metric_version,
        })
        store.finalize()
        wb.artifact(
            f"{args.study}-{spec.identity}", "training-run",
            [
                str(store.path / "run_spec.json"), str(store.path / "checkpoint.pt"),
                str(result), str(store.path / "evaluation.json"), str(store.path / "manifest.json"),
            ],
            metadata={"run_spec_hash": spec.identity, "arm": args.arm, "seed": args.seed},
            aliases=["latest", f"{args.arm}-s{args.seed}"],
        )
        atomic_write_json(active, {
            "schema_version": 1, "status": "complete", "study": args.study,
            "arm": args.arm, "seed": args.seed, "optimizer_updates": args.steps,
            "pid": os.getpid(), "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_spec_hash": spec.identity, "run_dir": store.path.as_posix(),
            "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
        })
        rhvd.log(f"RHVD_V2_DONE {args.arm} {spec.identity}")
    except Exception:
        atomic_write_json(active, {
            "schema_version": 1, "status": "failed", "study": args.study,
            "arm": args.arm, "seed": args.seed, "optimizer_updates": args.steps,
            "pid": os.getpid(), "failed_utc": datetime.now(timezone.utc).isoformat(),
            "run_spec_hash": spec.identity, "run_dir": store.path.as_posix(),
            "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
        })
        raise
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
