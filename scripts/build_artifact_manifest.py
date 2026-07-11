"""Build a SHA-256 index for explicitly named evidence files without scanning transient logs."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    files = []
    for raw in args.paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_write_json(args.out, {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": files,
    })
    print(f"wrote {args.out} with {len(files)} artifacts")


if __name__ == "__main__":
    main()
