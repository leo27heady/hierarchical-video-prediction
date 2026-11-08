import json

from scripts.validate_phasec_completion import _read_jsonl


def test_read_jsonl_accepts_windows_utf8_bom(tmp_path):
    path = tmp_path / "gpu_hours.jsonl"
    rows = [{"phase": "predictor-training", "gpu_hours": 1.25}, {"exit_code": 0}]
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8-sig")

    assert _read_jsonl(path) == rows
