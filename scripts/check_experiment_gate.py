"""Fail closed unless an experiment is authorized and all CPU/document gates exist."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.experiment_gate import check_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("--ledger", default="experiments/authorization.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = check_experiment(args.experiment, args.ledger, root=args.root)
    print(f"{result.experiment}: {result.status}")
    for failure in result.failures:
        print(f"  BLOCK: {failure}")
    result.require()
    print("  GATE PASS")


if __name__ == "__main__":
    main()
