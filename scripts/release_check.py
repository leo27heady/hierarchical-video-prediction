"""Run the complete CPU-only scientific and packaging release gate."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--wheel-dir", type=Path)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    python = sys.executable
    run(python, "scripts/validate_scientific_configs.py")
    run(python, "scripts/check_text_consistency.py", "--gate")
    run(python, "scripts/check_rft_consistency.py")
    run(python, "scripts/assemble_final_manuscripts.py")
    run(python, "-m", "compileall", "-q", "hvp", "hvp_rev")
    run(python, "-m", "hvp", "--version")
    if not args.skip_tests:
        run(python, "-m", "pytest", "-q")

    if args.wheel_dir:
        args.wheel_dir.mkdir(parents=True, exist_ok=True)
        run(python, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(args.wheel_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="hvp-wheel-") as directory:
            run(python, "-m", "pip", "wheel", ".", "--no-deps", "-w", directory)
            wheels = list(Path(directory).glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(f"expected one wheel, found {wheels}")
            print(f"WHEEL_BUILD_PASS {wheels[0].name}")
    print("RELEASE_CHECK_PASS")


if __name__ == "__main__":
    main()
