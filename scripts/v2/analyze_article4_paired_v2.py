"""Versioned descriptive paired analysis of frozen A4 per-clip arrays.

No winner reselection or noninferiority verdict. Run separately for each training
seed. Family size must be fixed before outcomes; Bonferroni percentile intervals
provide an approximate family adjustment, not a retrospective preregistration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def foreground_scores(squared_error, mask):
    """Return historical pixel-weighted and final per-clip means, plus empty count."""
    squared_error, mask = np.asarray(squared_error), np.asarray(mask, bool)
    if squared_error.shape != mask.shape or squared_error.ndim < 2 or squared_error.shape[0] == 0:
        raise ValueError('matching [clip,...pixel] shapes required')
    axes = tuple(range(1, mask.ndim))
    counts = mask.sum(axes)
    sums = (squared_error * mask).sum(axes)
    per_clip = sums / np.maximum(counts, 1)
    return dict(pixel_weighted=float(sums.sum()/max(counts.sum(),1)),
                clip_mean=float(per_clip.mean()), empty_clips=int((counts==0).sum()))


def primary(array):
    clips = np.asarray(array['clip'])
    horizons = np.asarray(array['horizon'])
    errors = np.asarray(array['foreground_mse'])
    if clips.shape != horizons.shape or clips.shape != errors.shape:
        raise ValueError('clip/horizon/metric shapes differ')
    select = horizons == 4
    ids, values = clips[select], errors[select]
    if ids.size == 0 or len(np.unique(ids)) != len(ids) or not np.array_equal(np.sort(ids), np.unique(clips)):
        raise ValueError('exactly one horizon-4 value per clip required')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite primary values')
    order = np.argsort(ids)
    return ids[order], values[order]


def paired(flat, hier, family_size, repetitions=10000, seed=260905):
    if family_size < 1 or repetitions < 100:
        raise ValueError('positive family size and at least 100 bootstrap repeats required')
    ids, a = primary(flat)
    other_ids, b = primary(hier)
    if not np.array_equal(ids, other_ids):
        raise ValueError('unpaired clip IDs')
    delta = a-b
    rng = np.random.default_rng(seed)
    means = np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(repetitions)])
    tail = 0.05/(2*family_size)
    return dict(schema_version=2, evidence_class='descriptive paired checkpoint analysis',
                horizon=4, clips=len(ids), flat_mean=float(a.mean()), hierarchy_mean=float(b.mean()),
                flat_minus_hierarchy=float(delta.mean()),
                adjusted_percentile_interval=np.quantile(means,[tail,1-tail]).tolist(),
                family_size=family_size, bootstrap_repetitions=repetitions, bootstrap_seed=seed,
                resampling_unit='clip within one fixed training seed',
                empty_mask_policy='upstream zero with denominator clamped to one; count unavailable in legacy arrays',
                noninferiority_verdict=None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flat',type=Path,required=True); p.add_argument('--hier',type=Path,required=True)
    p.add_argument('--family-size',type=int,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=paired(np.load(args.flat),np.load(args.hier),args.family_size)
    result['sources']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.flat,args.hier)}
    result['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
