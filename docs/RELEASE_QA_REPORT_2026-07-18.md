# Release and Manuscript QA Report — 18 July 2026

> **Superseded scope note — 19 July 2026.** This report records the 18 July review artifacts and
> scientific gates. It does not certify the later content-refactored Markdown, an institutional
> dissertation render, a journal-template article, or a clean-clone public release. A subsequent
> audit found visible linear-LaTeX/Markdown artifacts and table-pagination defects in the 18 July
> renders, developer-specific path defaults in research bridges, and bytecode/cache files in the
> old release manifest. The authoritative correction programme is
> `docs/CONTENT_FINALIZATION_EXECUTION_PLAN_2026-07-19.md`. The scientific evidence verdict remains
> unchanged; the earlier zero-formatting-finding and externally release-ready implications are
> withdrawn.

## Outcome

The current programme is release-ready at the research-content level. No further GPU experiment
is required for the binding fourth article or the dissertation claims. E9 is terminally skipped
under its post-freeze utility rule.

The authoritative deliverables are the English Markdown sources. The supplied DOCX and PDF files
are polished review editions for author review; they do not claim compliance with an as-yet
unspecified university or journal template.

## Scientific and source gates

- Scientific configuration/authorization validation: **PASS**; 10 configs, fail closed.
- Claim/text consistency: **PASS**; no projected-result shells remain.
- Reach–fidelity–trust consistency: **PASS** across 5 authoritative texts, 8 detailed chapters
  and the resource-synthesis document.
- Dissertation assembly: **PASS**; 54,178 words, 57 figure captions, 21 table captions,
  80 bound external references.
- Article assembly: **PASS**; 6,127 words.
- Local manuscript image links: **PASS**.
- Unresolved `[REF]`, `[UA-PASS]`, expected or pending markers: **0**.
- Remaining `[AUTHOR: ...]` markers: **12**, all administrative and indexed in
  `docs/dissertation/AUTHOR_INPUT_CHECKLIST.md`.
- Source scan for developer-specific absolute paths in Python/PowerShell/shell code: **PASS**.

## Code and package gates

- Python compilation: **PASS** for `hvp` and `hvp_rev`.
- CPU test suite: **83 passed**.
- Public CLI smoke test: **PASS**, version `hvp 0.3.0`.
- Wheel build: **PASS**,
  `hierarchical_video_prediction-0.3.0-py3-none-any.whl`.
- Historical development-only figure scripts require an explicit opt-in environment variable.
- The legacy GPU queue entry point is terminally disabled.

## Document and PDF verification

Both DOCX files were exported through Microsoft Word with structure tags enabled and rendered
through Poppler at 110 dpi.

| Artifact | Pages | Blank pages | Edge-touch pages | Low-text pages | Accessibility findings |
|---|---:|---:|---:|---:|---:|
| Article review edition | 23 | 0 | 0 | 0 | 0 high / 0 medium / 0 low |
| Dissertation review edition | 154 | 0 | 0 | 0 | 0 high / 0 medium / 0 low |

All 23 article pages and all 154 dissertation pages were inspected visually. Figures and tables
remain within the page geometry, captions are paired with their content, headings are consistent,
list numbering restarts correctly, and the hidden canonical source marker is absent from the
render. PDF text extraction confirms that no unresolved reference or projected-result marker is
present.

Equations are preserved as centered Cambria Math linear source in the review DOCX/PDF editions.
The Markdown contains the authoritative mathematical notation for later LaTeX or institutional
typesetting.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Article DOCX | `77df3b4b96ce59bde4b0c481bf327ff78a6c566dfa1e71a5709830f32fa00138` |
| Article PDF | `152e52a80d4fc019b41c3819597983fb353cb87c81de9d7e1c5e27549af57aa7` |
| Dissertation DOCX | `98bf88832d8d41c825bc6b810a279c2da1cf8881983b7d53952a46fded531f36` |
| Dissertation PDF | `52132f661c81b63342049b6507a78a457cc5bd8b6075d95b8c14aaaaec2dc73f` |
| Python wheel | `0cfc4b106422b59d714c6ecef8a327d98510dfc3b39f012ee40cc3036ba94928` |

## Author/institution boundary

The remaining work is not scientific refactoring:

1. supply the UDC and two research-project numbers;
2. bind article [A4], six conference records and final appendix letters/scans;
3. choose the target article venue;
4. apply the university/venue template and update pagination;
5. publish large content-addressed data/checkpoint artifacts if external one-command
   reproduction is required.
