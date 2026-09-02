"""Preflight, run, resume, and inspect the four-run OpenSTL confirmation queue."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def sha256(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("campaign schema_version must be 1")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("campaign must contain runs")
    identities = [run.get("run_id") for run in runs]
    if any(not identity for identity in identities) or len(identities) != len(set(identities)):
        raise ValueError("run IDs must be nonempty and unique")
    cells = [(run.get("dataset"), run.get("seed")) for run in runs]
    if len(cells) != len(set(cells)):
        raise ValueError("dataset/seed cells must be unique")
    if set(cells) != {("mmnist", 0), ("mmnist", 1), ("taxibj", 0), ("taxibj", 1)}:
        raise ValueError("final campaign must be exactly two datasets by seeds 0 and 1")
    return manifest


def git_state(root: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    return {"root": str(root), "commit": commit, "branch": branch, "clean": not bool(status), "status": status}


def gpu_state() -> dict:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        name, utilization, used, total, temperature = [part.strip() for part in output.split(",")]
        return {
            "available": True, "name": name, "utilization_percent": int(utilization),
            "memory_used_mib": int(used), "memory_total_mib": int(total),
            "temperature_c": int(temperature),
        }
    except Exception as error:
        return {"available": False, "error": repr(error)}


def prepared_data_paths(data_root: Path) -> dict:
    return {
        "mmnist_train_digits": data_root / "moving_mnist" / "train-images-idx3-ubyte.gz",
        "mmnist_validation": data_root / "moving_mnist" / "mnist_validation_seq_seed260902.npy",
        "mmnist_validation_manifest": data_root / "moving_mnist" / "mnist_validation_seq_seed260902.manifest.json",
        "mmnist_test": data_root / "moving_mnist" / "mnist_test_seq.npy",
        "taxibj_source": data_root / "taxibj" / "dataset.npz",
        "taxibj_X_train": data_root / "taxibj" / "memmap_v1" / "X_train.npy",
        "taxibj_Y_train": data_root / "taxibj" / "memmap_v1" / "Y_train.npy",
        "taxibj_X_test": data_root / "taxibj" / "memmap_v1" / "X_test.npy",
        "taxibj_Y_test": data_root / "taxibj" / "memmap_v1" / "Y_test.npy",
        "taxibj_manifest": data_root / "taxibj" / "memmap_v1" / "manifest.json",
    }


def preflight(manifest_path: Path, *, require_prepared_data: bool, require_idle_gpu: bool,
              check_wandb: bool, full_data_hash: bool = False) -> tuple[dict, bool]:
    manifest = load_manifest(manifest_path)
    hvp_root = Path(manifest["hvp_root"])
    openstl_root = Path(manifest["openstl_root"])
    data_root = Path(manifest["openstl_reference_data_root"])
    output_root = Path(manifest["output_root"])
    checks = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for name, path in (("hvp_root", hvp_root), ("openstl_root", openstl_root),
                       ("data_root", data_root), ("python", Path(manifest["python"]))):
        check(name, path.exists(), str(path))
    if hvp_root.exists():
        state = git_state(hvp_root)
        check("hvp_git_clean", state["clean"], state)
    if openstl_root.exists():
        state = git_state(openstl_root)
        check("openstl_git_clean", state["clean"], state)

    for run in manifest["runs"]:
        config = Path(run["config"])
        check(f"config:{run['run_id']}", config.is_file(), {
            "path": str(config), "sha256": sha256(config) if config.is_file() else None,
        })

    data = prepared_data_paths(data_root)
    data_identity = {}
    for name, path in data.items():
        required = require_prepared_data or name in {"mmnist_train_digits", "mmnist_test", "taxibj_source"}
        detail = {"path": str(path), "bytes": path.stat().st_size if path.is_file() else None}
        if path.is_file() and full_data_hash:
            detail["sha256"] = sha256(path)
            data_identity[name] = detail["sha256"]
        if required:
            check(f"data:{name}", path.is_file(), detail)
        else:
            checks.append({"name": f"data:{name}", "passed": path.is_file(),
                           "warning_only": True, "detail": detail})

    # Prepared manifests must describe the exact files used by the locked run.  A
    # cheap preflight validates sizes; the publication/execute preflight also
    # compares every recorded digest against the bytes on disk.
    manifest_specs = []
    if data["mmnist_validation_manifest"].is_file():
        record = json.loads(data["mmnist_validation_manifest"].read_text(encoding="utf-8"))
        manifest_specs.append(("mmnist_validation", data["mmnist_validation"], record))
    if data["taxibj_manifest"].is_file():
        records = json.loads(data["taxibj_manifest"].read_text(encoding="utf-8"))
        for record_name, data_name in (
            ("source", "taxibj_source"), ("X_train", "taxibj_X_train"),
            ("Y_train", "taxibj_Y_train"), ("X_test", "taxibj_X_test"),
            ("Y_test", "taxibj_Y_test"),
        ):
            if record_name in records:
                manifest_specs.append((data_name, data[data_name], records[record_name]))
    for name, path, record in manifest_specs:
        valid = path.is_file() and record.get("bytes") == path.stat().st_size
        detail = {
            "path": str(path), "recorded_bytes": record.get("bytes"),
            "actual_bytes": path.stat().st_size if path.is_file() else None,
            "recorded_sha256": record.get("sha256"),
        }
        if path.is_file() and full_data_hash:
            actual = data_identity.get(name) or sha256(path)
            data_identity[name] = actual
            detail["actual_sha256"] = actual
            valid = valid and record.get("sha256") == actual
        check(f"prepared_manifest:{name}", valid, detail)

    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    free_gib = usage.free / 1024 ** 3
    minimum = float(manifest["minimum_free_gib_after_projection"])
    # Conservative code/checkpoint/log projection before measured checkpoint sizes exist.
    projected_gib = 8.0
    check("output_free_space", free_gib - projected_gib >= minimum, {
        "free_gib": round(free_gib, 3), "projected_use_gib": projected_gib,
        "required_remaining_gib": minimum,
    })
    gpu = gpu_state()
    check("cuda_device", gpu.get("available", False), gpu)
    if require_idle_gpu and gpu.get("available"):
        idle = gpu["utilization_percent"] < 20 and gpu["memory_used_mib"] < 1500
        check("gpu_idle", idle, gpu)

    if check_wandb:
        command = [manifest["python"], "-c", "import wandb; print(wandb.Api(timeout=10).viewer)"]
        probe = subprocess.run(command, text=True, capture_output=True, timeout=20)
        check("wandb_authenticated", probe.returncode == 0, {
            "stdout": probe.stdout.strip(), "stderr": probe.stderr.strip(),
        })

    required_failures = [item for item in checks if not item["passed"] and not item.get("warning_only")]
    config_identity = {
        run["run_id"]: sha256(Path(run["config"]))
        for run in manifest["runs"] if Path(run["config"]).is_file()
    }
    repository_identity = {}
    if hvp_root.exists():
        repository_identity["hvp"] = git_state(hvp_root)["commit"]
    if openstl_root.exists():
        repository_identity["openstl"] = git_state(openstl_root)["commit"]
    lock_material = {
        "manifest_sha256": sha256(manifest_path),
        "repositories": repository_identity,
        "configs": config_identity,
        "data": data_identity if full_data_hash else None,
    }
    lock_fingerprint = None
    if full_data_hash and len(data_identity) == len(data):
        lock_fingerprint = hashlib.sha256(
            json.dumps(lock_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    report = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "full_data_hash": full_data_hash,
        "lock_material": lock_material,
        "lock_fingerprint": lock_fingerprint,
        "campaign_id": manifest["campaign_id"],
        "checked_at": time.time(),
        "checks": checks,
        "ready": not required_failures,
        "required_failures": [item["name"] for item in required_failures],
    }
    return report, not required_failures


def run_status(manifest: dict) -> list[dict]:
    root = Path(manifest["output_root"])
    rows = []
    for run in manifest["runs"]:
        folder = root / run["run_id"]
        result_path, state_path, failure_path = folder / "result.json", folder / "status.json", folder / "failure.json"
        if result_path.is_file():
            value = json.loads(result_path.read_text(encoding="utf-8"))
            status = value.get("status", "completed")
        elif state_path.is_file():
            value = json.loads(state_path.read_text(encoding="utf-8"))
            status = value.get("status", "unknown")
        elif failure_path.is_file():
            value = json.loads(failure_path.read_text(encoding="utf-8"))
            status = "failed"
        else:
            value, status = {}, "pending"
        checkpoint = folder / "checkpoints" / "last.ckpt"
        rows.append({
            "run_id": run["run_id"], "dataset": run["dataset"], "seed": run["seed"],
            "status": status, "resumable_checkpoint": checkpoint.is_file(),
            "elapsed_hours": value.get("elapsed_hours"),
        })
    return rows


def validate_authorization(manifest_path: Path, authorization_path: Path,
                           lock_fingerprint: str | None = None) -> dict:
    if not authorization_path.is_file():
        raise PermissionError(f"authorization ledger is missing: {authorization_path}")
    value = json.loads(authorization_path.read_text(encoding="utf-8"))
    if value.get("approved") is not True:
        raise PermissionError("authorization ledger is not approved")
    expected = sha256(manifest_path)
    if value.get("manifest_sha256") != expected:
        raise PermissionError("authorization does not match the current campaign manifest")
    if lock_fingerprint is not None and value.get("lock_fingerprint") != lock_fingerprint:
        raise PermissionError("authorization does not match the locked code/config/data fingerprint")
    return value


@contextmanager
def controller_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "controller.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"another controller owns {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": time.time()}, handle)
    try:
        yield
    finally:
        with suppress(FileNotFoundError):
            path.unlink()


def execute(manifest_path: Path, manifest: dict, arguments) -> int:
    if manifest.get("launch_authorized") is not True:
        raise PermissionError(
            "campaign is prelock; set launch_authorized only after bounded gates and Leonid's confirmation"
        )
    report, ready = preflight(
        manifest_path, require_prepared_data=True, require_idle_gpu=not arguments.allow_busy_gpu,
        check_wandb=arguments.check_wandb, full_data_hash=True,
    )
    atomic_json(Path(manifest["output_root"]) / "preflight.json", report)
    if not ready:
        print(json.dumps(report, indent=2))
        return 2
    if not report["lock_fingerprint"]:
        raise RuntimeError("publication preflight did not produce a complete lock fingerprint")
    authorization = validate_authorization(
        manifest_path, arguments.authorization, report["lock_fingerprint"]
    )
    output_root = Path(manifest["output_root"])
    worker = Path(manifest["hvp_root"]) / "scripts" / "v2" / "train_integrated_openstl.py"
    with controller_lock(output_root):
        for run in manifest["runs"]:
            folder = output_root / run["run_id"]
            result = folder / "result.json"
            if result.is_file():
                value = json.loads(result.read_text(encoding="utf-8"))
                if value.get("status") == "completed":
                    print(f"skip completed {run['run_id']}", flush=True)
                    continue
            folder.mkdir(parents=True, exist_ok=True)
            state = {
                "status": "running", "run_id": run["run_id"], "pid": None,
                "started_at": time.time(), "manifest_sha256": sha256(manifest_path),
                "authorization": authorization,
            }
            command = [
                manifest["python"], "-u", str(worker),
                "--config", run["config"],
                "--openstl", manifest["openstl_root"],
                "--data-root", manifest["openstl_reference_data_root"],
                "--output", str(folder),
                "--dataset", run["dataset"],
                "--seed", str(run["seed"]),
                "--run-id", run["run_id"],
                "--wandb-project", manifest["wandb_project"],
                "--wandb-group", manifest["wandb_group"],
                "--wandb-mode", arguments.wandb_mode,
                "--num-workers", str(arguments.num_workers),
                "--log-every", str(arguments.log_every),
                "--checkpoint-every", str(arguments.checkpoint_every),
            ]
            if arguments.no_wandb:
                command.append("--no-wandb")
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            with (folder / "stdout.log").open("a", encoding="utf-8") as stdout, \
                 (folder / "stderr.log").open("a", encoding="utf-8") as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=creationflags)
                state["pid"] = process.pid
                atomic_json(folder / "status.json", state)
                print(f"started {run['run_id']} pid={process.pid}", flush=True)
                try:
                    return_code = process.wait()
                except KeyboardInterrupt:
                    state.update({"status": "interrupted", "interrupted_at": time.time()})
                    atomic_json(folder / "status.json", state)
                    if os.name == "nt":
                        with suppress(Exception):
                            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                    else:
                        process.send_signal(signal.SIGINT)
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=60)
                    return 130
            state.update({
                "status": "completed" if return_code == 0 else "failed",
                "finished_at": time.time(), "return_code": return_code,
            })
            atomic_json(folder / "status.json", state)
            if return_code:
                print(f"failed {run['run_id']} rc={return_code}", flush=True)
                if not arguments.continue_on_failure:
                    return return_code
            else:
                print(f"completed {run['run_id']}", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("operation", choices=("preflight", "status", "run"))
    result.add_argument(
        "--campaign", type=Path,
        default=Path("experiments/v2/campaigns/integrated_openstl_confirmation_2026-09-02.json"),
    )
    result.add_argument("--authorization", type=Path, default=Path("experiments/v2/campaigns/integrated_openstl_confirmation.authorization.json"))
    result.add_argument("--allow-missing-prepared-data", action="store_true")
    result.add_argument("--allow-busy-gpu", action="store_true")
    result.add_argument("--check-wandb", action="store_true")
    result.add_argument("--full-data-hash", action="store_true")
    result.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    result.add_argument("--no-wandb", action="store_true")
    result.add_argument("--continue-on-failure", action="store_true")
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--log-every", type=int, default=50)
    result.add_argument("--checkpoint-every", type=int, default=1000)
    return result


def main() -> int:
    arguments = parser().parse_args()
    manifest_path = arguments.campaign.resolve()
    manifest = load_manifest(manifest_path)
    if arguments.operation == "status":
        print(json.dumps(run_status(manifest), indent=2))
        return 0
    if arguments.operation == "preflight":
        report, ready = preflight(
            manifest_path,
            require_prepared_data=not arguments.allow_missing_prepared_data,
            require_idle_gpu=False,
            check_wandb=arguments.check_wandb,
            full_data_hash=arguments.full_data_hash,
        )
        print(json.dumps(report, indent=2))
        return 0 if ready else 2
    return execute(manifest_path, manifest, arguments)


if __name__ == "__main__":
    raise SystemExit(main())
