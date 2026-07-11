"""Generate canonical configuration, evidence and prior-art tables from a JSON snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def table(rows, columns):
    head = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        cells = [str(row.get(key, "")).replace("|", "\\|") for key, _ in columns]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/evidence_snapshot_2026-07-11.json")
    parser.add_argument("--out", default="docs/draft/RFT_GENERATED_TABLES.md")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    text = [
        "# Generated reach–fidelity–trust tables",
        "",
        f"> Status: {data['status']}. Generated from `{args.input}`; pending values remain explicit.",
        "",
        "## Configuration and interface",
        "",
        table(data["configurations"], [
            ("arm", "Arm"), ("package", "Package"), ("reach", "Reach"),
            ("context_tokens", "Context tok"), ("target_tokens", "Target tok"),
            ("total_tokens", "Total tok"), ("target_interface", "Target interface"),
            ("status", "Status"),
        ]),
        "",
        "## Claim and evidence snapshot",
        "",
        table(data["claims"], [
            ("id", "Claim"), ("contrast", "Contrast"), ("seed", "Seed"),
            ("effect", "Effect"), ("interval", "95% interval"),
            ("analysis", "Analysis"), ("status", "Status"),
        ]),
        "",
        "## Prior-art positioning",
        "",
        table(data["prior_art"], [
            ("work", "Work"), ("memory", "Memory"), ("matched_budget", "Matched budget"),
            ("analytic_observability", "Observability"), ("dynamic_reentry", "Dynamic re-entry"),
            ("self_fed", "Self-fed"), ("role", "Role here"),
        ]),
        "",
    ]
    Path(args.out).write_text("\n".join(text), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
