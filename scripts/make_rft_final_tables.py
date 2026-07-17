"""Build the article/dissertation evidence tables from frozen result artifacts.

Missing future stages remain explicit.  No numeric result is copied from prose: every measured
cell below is read from an immutable analysis JSON and the output records those hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(relative: str, *, optional: bool = False) -> tuple[dict | None, Path]:
    path = ROOT / relative
    if not path.is_file():
        if optional:
            return None, path
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8")), path


def md_table(headers: list[str], rows: list[list[object]], align: set[int] | None = None) -> str:
    align = align or set()
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---:" if index in align else "---" for index in range(len(headers))) + "|"
    body = []
    for row in rows:
        cells = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def interval(row: dict, digits: int = 4) -> str:
    return f"[{row['lo']:+.{digits}f}, {row['hi']:+.{digits}f}]"


ARMS = {
    "hier-min": (48, 2, 1, 0),
    "flat-min": (0, 14, 1, 0),
    "hier-s2-min": (24, 2, 2, 0),
    "flat-short": (0, 8, 1, 0),
    "hier-long": (48, 6, 1, 0),
    "flat-budget": (0, 18, 1, 0),
    "strided-flat": (0, 6, 1, 11),
    "hier-c1": (48, 16, 1, 0),
    "flat-c1": (0, 28, 1, 0),
    "hier-c2": (48, 26, 1, 0),
    "flat-c2": (0, 38, 1, 0),
    "hier-rich": (48, 36, 1, 0),
    "flat-long": (0, 48, 1, 0),
}


def arm_accounting(arm: str) -> tuple[int, int, int, int]:
    coarse, recent_fine, stride, strided_fine = ARMS[arm]
    context = coarse * 16 + (recent_fine + strided_fine) * 64
    coarse_targets = 4 // stride if coarse > 0 and stride > 1 else 4
    target = coarse_targets * 16 + 4 * 64
    total = context + target
    if strided_fine:
        reach = recent_fine + 1 + 4 * (strided_fine - 1) + 1
    else:
        reach = max(coarse * stride, recent_fine)
    return reach, context, target, total


def configuration_section() -> str:
    package = {
        "hier-min": "spatial: dense coarse + 2 recent fine",
        "flat-min": "flat contiguous fine",
        "hier-s2-min": "spatiotemporal: stride-2 coarse + 2 fine",
        "flat-short": "flat contiguous fine",
        "hier-long": "spatial: dense coarse + 6 recent fine",
        "flat-budget": "flat contiguous fine",
        "strided-flat": "temporal: 11 stride-4 fine anchors + 6 fine",
        "hier-c1": "spatial: dense coarse + 16 fine",
        "flat-c1": "flat contiguous fine",
        "hier-c2": "spatial: dense coarse + 26 fine",
        "flat-c2": "flat contiguous fine",
        "hier-rich": "spatial: dense coarse + 36 fine",
        "flat-long": "flat contiguous fine",
    }
    status = {
        "hier-s2-min": "configuration effect; target interface differs",
        "strided-flat": "near-matched to hier-long (1,088 vs 1,152)",
        "hier-c1": "frontier member",
        "flat-c1": "frontier member",
        "hier-c2": "frontier + regime control",
        "flat-c2": "frontier + regime control",
        "hier-rich": "B8×60k",
        "flat-long": "B8×60k",
    }
    rows = []
    for arm in ARMS:
        reach, context, target, total = arm_accounting(arm)
        rows.append(
            [
                arm,
                package[arm],
                reach,
                f"{context:,}",
                f"{target:,}",
                f"{total:,}",
                status.get(arm, "B16×30k"),
            ]
        )
    return md_table(
        ["Arm", "Context package", "Reach", "Context tokens", "Target tokens", "Total tokens", "Qualification"],
        rows,
        {2, 3, 4, 5},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/draft/RFT_GENERATED_TABLES.md")
    parser.add_argument("--index", default="artifacts/rft_claim_artifact_index.json")
    args = parser.parse_args()

    e2, e2_path = read_json("artifacts/confirmation_e2/results.json")
    e4, e4_path = read_json("artifacts/e4b_head_package/results.json")
    e5, e5_path = read_json("artifacts/e5a_trust_replication/results.json")
    e7c, e7c_path = read_json("artifacts/e7c_sampler_nfe/results.json")
    cw_black, cw_black_path = read_json("artifacts/cwvae_black_baseline/results.json")
    e7a, e7a_path = read_json("artifacts/e7a_codec_robustness/results.json", optional=True)
    e8a, e8a_path = read_json("artifacts/e8a_pose_mosaic/results.json", optional=True)

    source_paths = [e2_path, e4_path, e5_path, e7c_path, cw_black_path]
    for value, path in ((e7a, e7a_path), (e8a, e8a_path)):
        if value is not None:
            source_paths.append(path)

    text = [
        "# Generated reach–fidelity evidence tables",
        "",
        "> Generated only from frozen result JSONs. Effects use the convention printed in each "
        "section; clip-cluster intervals condition on the trained model, while t intervals use "
        "training seeds as units.",
        "",
        "## Configuration and exact interface accounting",
        "",
        configuration_section(),
        "",
        "The temporal `strided-flat` package has 1,088 context tokens, not 1,152. Its 64-token "
        "deficit relative to `hier-long` was preregistered as favoring the temporal control; the "
        "comparison is near-matched rather than exact.",
        "",
        "## Untouched confirmation: mechanistic bands",
        "",
    ]
    e2_rows = []
    for row in e2["primary"]:
        if row["metric"] != "fg":
            continue
        e2_rows.append(
            [
                row["pair"],
                row["seed"],
                row["population"],
                row["records"],
                row["clusters"],
                f"{row['mean']:+.4f}",
                interval(row),
            ]
        )
    text += [
        md_table(
            ["Pair", "Seed", "Registered population", "Records", "Clips", "Effect", "Paired 95% CI"],
            e2_rows,
            {1, 3, 4, 5, 6},
        ),
        "",
        "Positive favors the longer-reach registered candidate. These pair-specific bands were "
        "frozen before the E2 holdout was opened.",
        "",
        "## Untouched confirmation: tested frontier family",
        "",
    ]
    frontier_rows = []
    for key, row in e2["frontier"]["contrasts"].items():
        frontier_rows.append([key, f"{row['mean']:+.4f}", interval(row)])
    text += [
        md_table(["Frozen contrast", "Effect", "Simultaneous 95% CI"], frontier_rows, {1, 2}),
        "",
        f"Studentized max-statistic clip-cluster family; critical value "
        f"{e2['frontier']['critical_value']:.3f}. No interpolation is performed between budgets.",
        "",
        "## Head × package by training seed",
        "",
    ]
    e4_rows = []
    for row in e4["seeds"]:
        e4_rows.append(
            [
                row["seed"],
                f"{row['diffusion']['mean']:+.4f}",
                interval(row["diffusion"]),
                f"{row['regression']['mean']:+.4f}",
                interval(row["regression"]),
                f"{row['interaction']['mean']:+.4f}",
                interval(row["interaction"]),
            ]
        )
    text += [
        md_table(
            ["Seed", "Diffusion", "Clip 95% CI", "Regression", "Clip 95% CI", "Interaction", "Clip 95% CI"],
            e4_rows,
            set(range(7)),
        ),
        "",
    ]
    seed_summary_rows = []
    for key in ("diffusion", "regression", "interaction"):
        row = e4["training_seed_summaries"][key]
        seed_summary_rows.append(
            [key, row["seeds"], f"{row['mean']:+.4f}", f"{row['sample_sd']:.4f}", interval(row)]
        )
    text += [
        md_table(
            ["Estimand", "Seeds", "Across-seed mean", "SD", "t95 interval"],
            seed_summary_rows,
            {1, 2, 3, 4},
        ),
        "",
        "Positive package effects favor temporally sparse full-resolution context. The stable "
        "diffusion result and sign-changing regression/interaction results must not be averaged "
        "into an equivalence claim.",
        "",
        "## Addressability versus self-fed reliability",
        "",
    ]
    e5_rows = []
    for row in e5["seeds"]:
        early = row["primary"]["early_benefit"]
        late = row["primary"]["late_inversion"]
        e5_rows.append(
            [
                row["seed"],
                f"{early['mean']:+.4f}",
                interval(early),
                f"{late['mean']:+.4f}",
                interval(late),
            ]
        )
    text += [
        md_table(
            ["Seed", "Early h1–24", "95% CI", "Late h61–256", "95% CI"],
            e5_rows,
            set(range(5)),
        ),
        "",
        f"Registered verdict: **{e5['decision']['verdict']}**. Positive favors the long-reach "
        "hierarchy; the late negative direction appears only at seed 0.",
        "",
        "## Sampler/NFE robustness family",
        "",
    ]
    e7c_rows = []
    label = {
        "b896-reach": "896 reach",
        "b1152-interior": "1,152 interior",
        "b1152-package-shape": "1,088 vs 1,152 package shape",
        "b2432-inversion": "2,432 inversion",
    }
    for row in e7c["primary"]:
        e7c_rows.append(
            [
                label[row["contrast"]],
                row["nfe"],
                row["sampler_seed"],
                row["total_nfe_per_clip"],
                f"{row['mean']:+.4f}",
                interval(row),
            ]
        )
    text += [
        md_table(
            ["Contrast", "NFE/step", "Sampler seed", "NFE/clip", "Effect", "Simultaneous 95% CI"],
            e7c_rows,
            {1, 2, 3, 4, 5},
        ),
        "",
        f"Registered verdict: **{e7c['decision']['verdict']}**.",
        "",
        "## Alternate-codec robustness",
        "",
    ]
    if e7a is None:
        text += ["Pending terminal E7a artifact.", ""]
    else:
        e7a_rows = [
            [
                row["codec"],
                row["contrast"],
                f"{row['mean']:+.4f}",
                interval(row),
            ]
            for row in e7a["primary"]
        ]
        text += [
            md_table(["Codec", "Contrast", "Effect", "Simultaneous 95% CI"], e7a_rows, {2, 3}),
            "",
            f"Registered verdict: **{e7a['decision']['verdict']}**. This is a whole-codec-seed "
            "intervention conditional on predictor seed 0.",
            "",
        ]
    text += ["## Privileged pose-indexed raw-memory reference", ""]
    if e8a is None:
        text += ["Pending terminal E8a artifact.", ""]
    else:
        e8a_rows = [
            [row["effect"], f"{row['mean']:+.4f}", interval(row)]
            for row in e8a["primary"]
        ]
        text += [
            md_table(["Effect", "Estimate", "Simultaneous 95% CI"], e8a_rows, {1, 2}),
            "",
            f"Registered verdict: **{e8a['decision']['verdict']}**. The mosaic receives exact "
            "poses and raw observed pixels; it is not token matched.",
            "",
        ]
    text += [
        "## Architecture/protocol positioning",
        "",
        md_table(
            ["Work", "Memory mechanism", "Matched budget", "Exact observability", "Self-fed", "Role"],
            [
                ["CW-VAE", "clocked hierarchical latent state", "no", "no", "yes", "temporal-hierarchy lineage"],
                ["FDM", "flexible diffusion history conditioning", "no", "no", "yes", "diffusion-video baseline lineage"],
                ["TECO", "temporally consistent latent transformer", "no", "no", "yes", "long-horizon generation context"],
                ["WorldPack", "compressed trajectory packing + retrieval", "no", "no", "yes", "modern memory/compression comparator"],
                ["CaR / SceneFly", "compression + viewpoint-conditioned retrieval", "no", "annotated geometry", "yes", "modern retrieval comparator"],
                ["MemLearner / GIM-World", "learned query or geometry-aware implicit memory", "no", "annotated or distilled geometry", "yes", "learned-memory comparators"],
                ["MBench / MemoBench", "benchmark instruments", "n/a", "task dependent", "model dependent", "memory-evaluation context"],
                ["Self Forcing", "rollout-distribution training", "no", "no", "explicitly targeted", "self-fed mismatch intervention"],
                ["This work", "flat/spatial/temporal context packages", "yes where stated", "analytic LoopMom labels", "yes", "controlled reach–fidelity mechanism study"],
            ],
        ),
        "",
        "Numerical cross-paper ranking is intentionally omitted because dataset, decoder, context, "
        "training horizon, rollout, NFE and metrics are not protocol-equivalent.",
        "",
        "## Moving-MNIST full-frame metric sanity diagnostic",
        "",
    ]
    cw_rows = []
    for window_id, black_ssim in cw_black["black"]["ssim"]["windows"].items():
        cw_ssim = cw_black["cwvae"]["ssim"]["windows"][window_id]
        black_psnr = cw_black["black"]["psnr"]["windows"][window_id]
        cw_psnr = cw_black["cwvae"]["psnr"]["windows"][window_id]
        cw_rows.append([
            (
                f"+{black_ssim['start_frame_inclusive']}–"
                f"{black_ssim['stop_frame_inclusive']}"
            ),
            f"{cw_ssim['mean']:.4f}",
            f"{black_ssim['mean']:.4f}",
            f"{cw_psnr['mean']:.3f}",
            f"{black_psnr['mean']:.3f}",
        ])
    text += [
        md_table(
            ["Horizon", "CW-VAE SSIM", "Black SSIM", "CW-VAE PSNR", "Black PSNR"],
            cw_rows,
            {1, 2, 3, 4},
        ),
        "",
        (
            f"Same archived population: {cw_black['protocol']['clips']} clips, "
            f"{cw_black['protocol']['context']} context frames and "
            f"{cw_black['protocol']['horizon']} open-loop targets. CW-VAE is best-of-"
            f"{cw_black['protocol']['cwvae_samples']} by clip-average SSIM; black is "
            "deterministic. This is a background-dominance diagnostic, not a "
            "budget-matched model comparison, and the archived mean curves do not support "
            "paired inference."
        ),
        "",
        "## Artifact binding",
        "",
    ]
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_paths}
    text += [md_table(["Frozen result artifact", "SHA-256"], [[key, value] for key, value in hashes.items()]), ""]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(text), encoding="utf-8")
    index = {
        "schema_version": 1,
        "status": "SYNCHRONIZED" if e7a is not None and e8a is not None else "AWAITING_LATER_STAGES",
        "article_claims": {
            "reach_mechanism": "artifacts/confirmation_e2/results.json",
            "tested_frontier": "artifacts/confirmation_e2/results.json",
            "head_package_heterogeneity": "artifacts/e4b_head_package/results.json",
            "self_fed_negative_replication": "artifacts/e5a_trust_replication/results.json",
            "sampler_nfe_robustness": "artifacts/e7c_sampler_nfe/results.json",
            "moving_mnist_metric_sanity": "artifacts/cwvae_black_baseline/results.json",
            "codec_robustness": (
                "artifacts/e7a_codec_robustness/results.json" if e7a is not None else None
            ),
            "pose_indexed_reference": (
                "artifacts/e8a_pose_mosaic/results.json" if e8a is not None else None
            ),
        },
        "sha256": hashes,
    }
    index_path = ROOT / args.index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} and {index_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
