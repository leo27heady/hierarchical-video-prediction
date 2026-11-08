"""Prepare publication-safe validation/cache files for the integrated model.

This utility is intentionally explicit and is not called by the training loader.
TaxiBJ conversion temporarily reads one legacy float64 NPZ member at a time;
run it only when sufficient host RAM and disk are available.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

OPENSTL_ROOT = Path(__file__).resolve().parents[2]
if str(OPENSTL_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENSTL_ROOT))

from openstl.datasets.dataloader_moving_mnist import MovingMNIST


def sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_memmap(path: Path, dtype, shape):
    temporary = path.with_suffix(path.suffix + '.partial')
    if temporary.exists():
        temporary.unlink()
    array = np.lib.format.open_memmap(temporary, mode='w+', dtype=dtype, shape=shape)
    return temporary, array


def prepare_mmnist(root: Path, clips: int, seed: int) -> dict:
    output = root / 'moving_mnist' / f'mnist_validation_seq_seed{seed}.npy'
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    random.seed(seed)
    np.random.seed(seed)
    dataset = MovingMNIST(
        root=str(root), is_train=True, data_name='mnist',
        n_frames_input=10, n_frames_output=10, image_size=64,
        num_objects=[2], use_augment=False,
    )
    temporary, destination = atomic_memmap(output, np.uint8, (20, clips, 64, 64))
    for index in range(clips):
        inputs, targets = dataset[index]
        sequence = np.rint(
            np.concatenate((inputs.numpy(), targets.numpy()), axis=0)[:, 0] * 255.0
        ).clip(0, 255).astype(np.uint8)
        destination[:, index] = sequence
        if (index + 1) % 100 == 0:
            destination.flush()
            print(f'Moving-MNIST validation {index + 1}/{clips}', flush=True)
    destination.flush()
    del destination
    os.replace(temporary, output)
    record = {
        'path': str(output), 'shape': [20, clips, 64, 64], 'dtype': 'uint8',
        'bytes': output.stat().st_size, 'sha256': sha256(output), 'seed': seed,
    }
    manifest = output.with_suffix('.manifest.json')
    manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def prepare_taxibj(root: Path, chunk: int) -> dict:
    source = root / 'taxibj' / 'dataset.npz'
    destination_root = root / 'taxibj' / 'memmap_v1'
    destination_root.mkdir(parents=True, exist_ok=True)
    records = {}
    with np.load(source) as archive:
        for name in ('X_train', 'Y_train', 'X_test', 'Y_test'):
            output = destination_root / f'{name}.npy'
            if output.exists():
                raise FileExistsError(f'refusing to overwrite {output}')
            original = archive[name]
            temporary, destination = atomic_memmap(output, np.float32, original.shape)
            for begin in range(0, len(original), chunk):
                end = min(begin + chunk, len(original))
                destination[begin:end] = original[begin:end].astype(np.float32, copy=False)
                destination.flush()
                print(f'TaxiBJ {name} {end}/{len(original)}', flush=True)
            destination.flush()
            del destination
            os.replace(temporary, output)
            records[name] = {
                'path': str(output), 'shape': list(original.shape), 'dtype': 'float32',
                'bytes': output.stat().st_size, 'sha256': sha256(output),
            }
            del original
            gc.collect()
    records['source'] = {
        'path': str(source), 'bytes': source.stat().st_size, 'sha256': sha256(source),
    }
    manifest = destination_root / 'manifest.json'
    manifest.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--prepare', choices=('mmnist', 'taxibj', 'both'), default='both')
    parser.add_argument('--mmnist-clips', type=int, default=2000)
    parser.add_argument('--mmnist-seed', type=int, default=260902)
    parser.add_argument('--taxibj-chunk', type=int, default=128)
    args = parser.parse_args()
    result = {}
    if args.prepare in {'mmnist', 'both'}:
        result['moving_mnist_validation'] = prepare_mmnist(
            args.data_root, args.mmnist_clips, args.mmnist_seed
        )
    if args.prepare in {'taxibj', 'both'}:
        result['taxibj_cache'] = prepare_taxibj(args.data_root, args.taxibj_chunk)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
