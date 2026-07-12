"""Create a hash-bound JSON attestation for a reviewed gate subject."""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    subject = Path(args.subject)
    if not subject.is_file():
        raise FileNotFoundError(subject)
    source_commit = args.source_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    atomic_write_json(args.out, {
        "schema_version": 1,
        "status": args.status,
        "subject": subject.as_posix(),
        "subject_sha256": sha256_file(subject),
        "source_commit": source_commit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"wrote hash-bound attestation {args.out}")


if __name__ == "__main__":
    main()
