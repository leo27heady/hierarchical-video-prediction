"""Command-line entry point for safe inspection of the research package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .experiments import check_experiment


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser without causing side effects."""
    parser = argparse.ArgumentParser(
        prog="hvp",
        description="Inspect the hierarchical-video-prediction release and experiment gates.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate", help="check an experiment authorization without launching")
    gate.add_argument("experiment")
    gate.add_argument("--ledger", default="experiments/authorization.json")
    gate.add_argument("--root", default=".")
    gate.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a read-only CLI command and return its process status."""
    args = build_parser().parse_args(argv)
    if args.command == "gate":
        result = check_experiment(
            args.experiment,
            Path(args.ledger),
            root=Path(args.root),
        )
        payload = {
            "experiment": result.experiment,
            "status": result.status,
            "launchable": result.launchable,
            "failures": list(result.failures),
        }
        if args.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            verdict = "PASS" if result.launchable else "BLOCKED"
            print(f"{verdict}: {result.experiment} ({result.status})")
            for failure in result.failures:
                print(f"- {failure}")
        return 0 if result.launchable else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
