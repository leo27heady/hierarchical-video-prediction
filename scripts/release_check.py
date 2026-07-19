"""Run the complete CPU-only scientific and packaging release gate."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def validate_installed_wheel(wheel: Path) -> None:
    """Install the wheel into an isolated target and exercise it outside the source tree."""
    with tempfile.TemporaryDirectory(prefix="hvp-install-") as directory:
        target = Path(directory) / "site-packages"
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel.resolve()),
        )
        target_literal = repr(str(target.resolve()))
        run(
            sys.executable,
            "-I",
            "-c",
            (
                f"import pathlib, sys; sys.path.insert(0, {target_literal}); import hvp; "
                f"assert pathlib.Path(hvp.__file__).is_relative_to(pathlib.Path({target_literal})); "
                "from hvp.context import ArmSpec; "
                "assert ArmSpec(name='smoke', coarse_context_slots=48, "
                "fine_recent_slots=2).context_tokens == 896; "
                "print('CLEAN_WHEEL_IMPORT_PASS', hvp.__version__)"
            ),
        )
        run(
            sys.executable,
            "-I",
            "-c",
            (
                f"import runpy, sys; sys.path.insert(0, {target_literal}); "
                "sys.argv=['hvp', '--version']; runpy.run_module('hvp', run_name='__main__')"
            ),
        )


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

    with tempfile.TemporaryDirectory(prefix="hvp-wheel-") as directory:
        build_dir = Path(directory)
        run(python, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(build_dir))
        wheels = list(build_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {wheels}")
        wheel = wheels[0]
        print(f"WHEEL_BUILD_PASS {wheel.name}")
        validate_installed_wheel(wheel)
        if args.wheel_dir:
            args.wheel_dir.mkdir(parents=True, exist_ok=True)
            copied = Path(shutil.copy2(wheel, args.wheel_dir / wheel.name))
            print(f"WHEEL_COPIED {copied}")
    print("RELEASE_CHECK_PASS")


if __name__ == "__main__":
    main()
