# Generated scientific gates

Gate marker files are generated locally after the corresponding validation succeeds and are not
versioned.  Their absence is intentionally blocking.

- `cpu_validation.txt` — complete CPU suite and protocol validators passed for a recorded commit.
- `confirmation_protocol_frozen.txt` — untouched-confirmation endpoints and analysis were frozen.
- `phase_c_classifier_validated.txt` — classifier validation passed on clean, translated,
  overlapping, and reconstruction-domain crops.
- `phase_c_protocol_frozen.txt` — Phase C v2 protocol and equivalence policy were frozen.

A marker is evidence of a check, not authorization by itself.  The experiment's status in
`experiments/authorization.json` must also be launchable.
