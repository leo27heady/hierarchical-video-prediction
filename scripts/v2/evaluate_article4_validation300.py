"""Evaluate the frozen A4 shortlist on all development-validation clips, once.

No selection or confirmation read is performed. The selection ledger is the
existing finalizer's development-only decision. Outputs live outside the live tree.
"""
from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def shortlist(ledger):
    rows = {}
    for role in ('best_flat_1792', 'best_hierarchy_1792', 'memory_pareto_within_5pct'):
        for row in ledger['roles'][role]['runs']:
            rows[row['config_hash']] = row
    if not 2 <= len(rows) <= 6:
        raise ValueError('VAL300 requires at most three frozen two-seed configurations')
    return list(rows.values())


def validation_batches(clip_ids, batch_size):
    import numpy as np
    if len(clip_ids) != 300 or len(set(clip_ids)) != 300 or batch_size < 1:
        raise ValueError('expected 300 distinct validation clip IDs and a positive batch')
    for offset in range(0, 300, batch_size):
        ids = np.asarray(clip_ids[offset:offset + batch_size], dtype=np.int64)
        yield np.pad(ids, (0, batch_size - len(ids)), mode='edge'), len(ids)


def main(args):
    import numpy as np
    import torch
    from scripts.v2 import finalize_article4_campaign as f
    f.ROOT = args.a4_root.resolve()
    os.chdir(f.ROOT)  # Legacy AE paths are relative to the immutable live repository.
    ledger_path = f.ROOT / 'artifacts/v2/article4_primary_30k/post_training/selection_ledger.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    if ledger['status'] != 'locked_from_development_validation':
        raise ValueError('selection must already be frozen')
    args.output.mkdir(parents=True, exist_ok=True)
    records = shortlist(ledger)
    contract = {'selection_sha256': f.file_sha256(ledger_path), 'runs': records,
                'primary_horizon': 4, 'sampling_seed': 260907,
                'metric': 'per-clip foreground MSE; threshold >0.5; empty mask=0',
                'role': 'validation robustness only; no reselection or held-out claim',
                'evaluator_sha256': f.file_sha256(Path(__file__))}
    f.write_immutable_json(args.output / 'contract.json', contract)
    summaries = []
    for record in records:
        spec = f.load_campaign_run_spec(f.ROOT / record['config'])
        checkpoint = Path(record['checkpoint'])
        if not checkpoint.is_absolute():
            checkpoint = f.ROOT / checkpoint
        if f.file_sha256(checkpoint) != record['checkpoint_sha256']:
            raise ValueError('checkpoint identity changed')
        output = args.output / spec.experiment_id
        output.mkdir(exist_ok=True)
        result = output / 'summary.json'
        if result.exists():
            summaries.append(json.loads(result.read_text()))
            continue
        ae = f.hier3.load_top(spec.ae_path, spec.top_ae_path) if spec.top_ae_path else f.hier_convae.load(spec.ae_path)
        loaded = f.load_trb_development(ae, f.ROOT / spec.dataset_manifest)
        if spec.top_ae_path:
            f.hier3.attach_top(loaded.data, ae)
        clip_ids = np.asarray(loaded.manifest['split']['validation_indices'])
        if len(clip_ids) != 300:
            raise ValueError('expected exactly 300 validation clips')
        net, wrapper = f.build_model(spec, checkpoint)
        torch.manual_seed(260907)
        np.random.seed(260907)
        batcher = f.PackedWindowBatcher(loaded.data, spec.plan, batch_size=spec.validation_batch_size,
                                      seed=260907, device=f.legacy.DEV, context_anchor=spec.context_anchor)
        pixel, foreground, counts = [], [], []
        with torch.no_grad():
            for ids, n in validation_batches(clip_ids, spec.validation_batch_size):
                batch = batcher.sample(split='validation', include_frames=True, clip_indices=ids,
                                       starts=np.full(len(ids), spec.context_anchor, dtype=np.int64))
                sampled = wrapper.sample(batch['ctx_c'], batch['ctx_f'], batch['actions'],
                                         n_steps=spec.validation_nfe, **f.forward_kwargs(batch))[1]
                decoded = f.decode01(ae, sampled.reshape(-1, *sampled.shape[2:])).reshape(len(ids), 4, 1, 64, 64)
                target = batch['gt_frames_01']
                error = (decoded - target).square()
                mask = target > .5
                count = mask.sum(dim=(2, 3, 4))
                pixel.append(error.mean(dim=(2, 3, 4)).cpu().numpy()[:n])
                foreground.append(((error * mask).sum(dim=(2, 3, 4)) / count.clamp_min(1)).cpu().numpy()[:n])
                counts.append(count.cpu().numpy()[:n])
        fg = np.concatenate(foreground)
        if not np.isfinite(fg).all():
            raise ValueError('nonfinite VAL300 result')
        np.savez_compressed(output / 'per_clip_horizon_metrics.npz', clip_id=clip_ids,
                            horizon=np.arange(1, 5), pixel_mse=np.concatenate(pixel),
                            foreground_mse=fg, foreground_pixels=np.concatenate(counts))
        summary = {'run_id': spec.experiment_id, 'clips': 300, 'horizon4_foreground_mse': float(fg[:, 3].mean()),
                   'checkpoint_sha256': record['checkpoint_sha256'], 'status': 'completed'}
        f.atomic_json(result, summary)
        summaries.append(summary)
        del batcher, loaded, ae, net, wrapper, batch, sampled, decoded, target, error, mask
        torch.cuda.empty_cache()
    f.atomic_json(args.output / 'result.json', {'status': 'completed', 'runs': summaries, 'contract': contract})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a4-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
