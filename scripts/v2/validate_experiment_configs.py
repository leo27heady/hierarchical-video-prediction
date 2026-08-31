"""Validate every repository v2 experiment template and print immutable identities."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.config import load_experiment_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("experiments/v2/configs"),
    )
    args = parser.parse_args()

    paths = sorted(args.root.rglob("*.json"))
    if not paths:
        raise SystemExit(f"no JSON configs found under {args.root}")

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for path in paths:
        config = load_experiment_config(path)
        if config.experiment_id in seen_ids:
            raise ValueError(f"duplicate experiment_id: {config.experiment_id}")
        if config.content_hash in seen_hashes:
            raise ValueError(f"duplicate resolved config: {path}")
        seen_ids.add(config.experiment_id)
        seen_hashes.add(config.content_hash)
        print(
            f"{config.experiment_id} "
            f"hash={config.content_hash[:12]} "
            f"tokens={config.context_tokens} reach={config.nominal_reach} "
            "launch=BLOCKED"
        )

    print(f"PASS: {len(paths)} v2 templates validated; all fail closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
