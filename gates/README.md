# Generated scientific gates

Gate reports and hash-bound JSON attestations are generated after the corresponding validation
succeeds. JSON reports/attestations are versioned as scientific audit evidence; legacy free-text
markers remain ignored. Their absence or a subject-hash mismatch is intentionally blocking.

- `e2_cpu_validation_report.json` — complete CPU suite and protocol validators passed on a clean
  recorded commit.
- `cpu_validation.json` — hash-bound attestation of that report.
- `confirmation_protocol_frozen.json` — hash-bound untouched-confirmation protocol attestation.
- `e2_design_frozen.json` — hash-bound E2 run-matrix attestation.
- `cpu_validation.txt` — legacy human-readable mirror, ignored by Git and not sufficient alone.
- `phase_c_classifier_validated.txt` — classifier validation passed on clean, translated,
  overlapping, and reconstruction-domain crops.
- `phase_c_protocol_frozen.txt` — Phase C v2 protocol and equivalence policy were frozen.

A marker is evidence of a check, not authorization by itself. The experiment's status in
`experiments/authorization.json` must also be launchable.
