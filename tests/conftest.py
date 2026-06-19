"""Global test configuration (ported verbatim from SEED-Voken / the continuous-joint-blt branch).

Windows CPU runs of the full suite intermittently die with a native access
violation inside MHA softmax; isolated files always pass. Mitigations, all
required together:
- single-threaded torch (multi-threaded OpenMP makes the crash far more likely)
- KMP_DUPLICATE_LIB_OK guard (duplicate libiomp5md.dll from MKL + torch)
- gc between tests (the crash correlates with accumulated allocator pressure)
The env vars must be set before torch initializes its OpenMP runtime, hence
the import order here.
"""

import gc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


try:
    import psutil

    _PROC = psutil.Process()
except Exception:  # pragma: no cover
    _PROC = None

# Whole-suite RSS budget. If a test trips this, shrink its stack — do NOT raise it.
_RSS_BUDGET_BYTES = 12 * 1024**3
_budget_tripped = False
_peak_raisers: list = []


def _peak_wset() -> int:
    if _PROC is None:
        return 0
    info = _PROC.memory_info()
    return getattr(info, "peak_wset", info.rss)


@pytest.fixture(autouse=True)
def _collect_garbage_and_check_rss(request):
    peak_before = _peak_wset()
    yield
    gc.collect()
    global _budget_tripped
    if _PROC is None:
        return
    peak_after = _peak_wset()
    if peak_after - peak_before > 100 * 1024**2:
        _peak_raisers.append((request.node.nodeid, peak_after - peak_before, peak_after))
    if _budget_tripped:
        return
    rss = _PROC.memory_info().rss
    if rss >= _RSS_BUDGET_BYTES:
        _budget_tripped = True
        raise AssertionError(
            f"{request.node.nodeid} left RSS at {rss / 1024**3:.1f} GiB "
            f"(budget {_RSS_BUDGET_BYTES / 1024**3:.0f} GiB) — shrink the test stack"
        )


def pytest_terminal_summary(terminalreporter):
    if not _peak_raisers:
        return
    terminalreporter.write_sep("-", "peak-RSS raisers (>100 MiB lifetime-peak increase)")
    for nodeid, delta, peak in sorted(_peak_raisers, key=lambda x: -x[1])[:10]:
        terminalreporter.write_line(
            f"+{delta / 1024**3:5.2f} GiB -> peak {peak / 1024**3:5.2f} GiB  {nodeid}"
        )
