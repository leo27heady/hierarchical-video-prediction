## Summary
Adds uncertainty/entropy maps, a 2→8→32 (128px) hierarchy, an explicit-motion ("warp") predictor head, and a deep, evidence-grounded diagnosis of the single-step prediction blob — plus review-driven correctness/robustness fixes. Full suite green.

## What's in here
- **Entropy/uncertainty maps** — `hvp/eval/uncertainty.py`, `scripts/uncertainty_map.py`, and an in-training hook saving B&W maps (white=certain / black=uncertain) to `<out>/val_strips/`.
- **2→8→32 (128px)** — `configs/so3_128.yaml` (verified stage shapes `(32,8,2)`), `scripts/gen_data_so3_128.py`.
- **Warp head** — per-cell flow + `grid_sample` of the detached current latent + residual (`predictor/cascade.py`, `heads.py`).
- **Working recipe + diagnostics** — `configs/moderate_z.yaml`, `scripts/gen_data_simple_z.py` (`--preset`, `--seq-len`), `scripts/debug_prediction.py`, `scripts/probe_128.py`.
- **Robustness** — honest pixel metrics (mean+clamp, changed-region, oracle ceiling, BOT-only), gradient accumulation, NaN stall canary, reshuffling loader, stamped+pruned+atomic checkpoints + wall-clock timer, topology guards.

## Key finding
The blob is the **BOT next-latent prediction mean-regressing, scaling with per-frame motion**; the tokenizer/decode are fine (oracle decode is sharp). It works at moderate motion; the warp head is marginal; `pred_bot+current-coarse` is a free +1–2 dB; fast/SO(3) is a **motion-inference** problem. See `docs/PREDICTION_FINDINGS.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
