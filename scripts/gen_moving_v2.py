"""Generate versioned Moving-MNIST data with persistent object metadata."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.data.moving_mnist import generate_moving_mnist, load_idx, save_moving_mnist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="official MNIST idx3 gzip")
    parser.add_argument("--labels", required=True, help="matching official MNIST idx1 gzip")
    parser.add_argument("--source-split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--purpose", choices=["training", "development", "phase_c_v2_holdout"], required=True
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--objects", type=int, default=2)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    images, labels = load_idx(args.images, args.labels)
    bundle = generate_moving_mnist(
        images, labels, n_clips=args.n, frames=args.frames, image_size=args.size,
        objects=args.objects, seed=args.seed, step=args.step, source_split=args.source_split,
    )
    manifest = save_moving_mnist(
        bundle, args.out, purpose=args.purpose, generator_commit=args.generator_commit,
        source_images_path=args.images, source_labels_path=args.labels,
    )
    print(f"saved {args.out} with {manifest['artifacts']['metadata']['path']}")
    print(f"dataset manifest sha256: {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
