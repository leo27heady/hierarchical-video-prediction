# Content Finalization Report — 19 July 2026

## Verdict

The scientific programme is closed for the present article and dissertation: the repository
contains enough evidence, no additional GPU experiment is scientifically required, and E9 remains
terminally skipped.

The **Markdown content is now a defensible final scientific source**, but “final” must be qualified:

- the fourth article is ready for adaptation to a selected journal or conference, not yet a
  venue-formatted submission;
- the dissertation is ready for institutional assembly and author verification, not yet a
  university-compliant submitted document;
- the public Python surface builds and validates, but the large artifact/checkpoint archive is
  not yet publicly downloadable and the frozen research internals are not a v1-quality API.

These remaining conditions are administrative, typesetting, distribution, or external-validity
boundaries. They do not conceal a missing central experiment.

## What changed

### Fourth article

The authoritative source is
`docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md`.

- The article now opens with the fixed-budget memory problem and the untouched result.
- The abstract is 278 words.
- The central frontier is discrete and honest: hierarchy at 896 and 1,152; 1,792 unresolved; flat
  at 2,432 and 3,072.
- The 2.67–3.43-fold number is explicitly a context-token reduction for nominal reach, not an
  equal-performance or equal-compute claim.
- The package definition now includes representation level, temporal stride, slot count, and
  anchor lag. Context, target, and total tokens are distinguished.
- The diffusion objective now matches the executed v-parameterized implementation.
- Clip-cluster inference, training-seed uncertainty, and simultaneous comparison families are
  distinguished mathematically.
- E6 reports both relative effects and difficult absolute performance.
- The failed trust-boundary replication remains a negative result rather than a title-level law.
- Related work now positions MilliVid as the closest hierarchical-latent comparator and includes
  current memory/retrieval benchmarks and models.
- The bibliography contains 46 used references. The recent 2025–2026 entries were checked against
  their primary arXiv records.

The storyline is strong because every major result now answers one of three questions: whether the
past is reachable, whether useful detail survives, and whether stored information remains usable
in self-fed rollout. Robustness studies narrow that claim rather than competing with it.

### Dissertation

The authoritative editable sources remain under `docs/dissertation/rewrite/`; generated bound
files are under `docs/dissertation/final/`.

- The front-matter abstract was reduced to a genuine dissertation abstract.
- The Introduction now follows the conventional structure of the supplied older dissertation
  `C:/Users/leoni/Downloads/LKulyk_dissertation_v2.docx`: relevance; programme connection; aim and
  tasks; object and subject; methods; scientific novelty; practical significance; reliability;
  personal contribution; approbation and publications; structure. Its outdated scientific scope
  was not restored.
- The systems thesis is explicit:

  \[
  M_{\mathrm{system}}=
  M_{\mathrm{parameters}}+
  M_{\mathrm{optimizer}}+
  M_{\mathrm{context/state}}+
  M_{\mathrm{activations}}+
  M_{\mathrm{transient}}.
  \]

- Context packaging and reversible computation are presented as separately measured interventions
  on different memory terms.
- Chapter 4 was rebuilt around research questions and operating regions rather than experiment
  chronology. The old detailed campaign narrative is preserved as
  `docs/dissertation/rewrite/appendices/CHAPTER4_CAMPAIGN_ARCHIVE.md`.
- General Conclusions now map to the seven research tasks without repeating a second full Results
  chapter.
- Chapters 1–2 were red-teamed for inherited categorical language. Claims that a mask guarantees
  physical plausibility, that energy universally detects anomalies, that every complex system has
  one hierarchy, or that RoPE-ND forces correct routing were removed.
- RoPE-ND is now correctly scoped as a separate [A1] representation result. The final Chapter 3
  predictor uses learned level/role/frame embeddings and joint attention; its hierarchy effects
  are not attributed to RoPE-ND or reversible rollout training.
- The assembled manuscript contains 38,069 words, 42 figure captions, 13 table captions, and 123
  bound external references.
- No unresolved scientific reference markers or projected/pending result shells remain.

### References

The reference expansion is sufficient for the current claims:

- article: 46 references;
- dissertation: 123 bound external references, in addition to the applicant-publication block.

Increasing the dissertation mechanically to 200 references would not improve it by itself.
Additional references should be added only for a new claim, a venue-requested survey expansion, or
a field-specific committee request. The current priority should be bibliographic verification and
institutional formatting, not a numerical quota.

### Release-facing corrections

- Executable dataset/reference paths now use environment variables or repository-relative
  locations rather than developer-specific defaults.
- The source manifest generator excludes bytecode, caches, rendered review files, and wheels and
  can require a clean tracked source state.
- The legacy `paired_ci` helper now requires clip labels; frame-IID fallback is no longer silently
  available.
- Documentation distinguishes the supported `hvp.*` API from frozen `hvp_rev`, `hvp_blt`, and
  `hvp_dlh` distribution dependencies.
- The release gate now builds the wheel, installs it to an isolated target, and exercises the
  installed public import and CLI outside the source tree.
- The 18 July DOCX/PDF and page-level QA statements are labelled historical; they do not certify
  the 19 July content.

## Verification

The final validation used the repository virtual environment:

- 10 scientific configurations and the fail-closed authorization ledger validated;
- text-consistency gate passed with zero projected-result shells;
- article/dissertation reach–fidelity consistency gate passed;
- reference binding and Markdown assembly passed;
- 83 CPU tests passed;
- Python compilation passed;
- `hvp 0.3.0` source CLI passed;
- wheel build passed;
- isolated installed-wheel import passed;
- isolated installed-wheel CLI passed.

An earlier attempt with the system Python failed because that interpreter did not contain SciPy.
This was an environment-selection error, not suppressed as a passing run; the documented command
uses the repository virtual environment.

## What still prevents the unqualified word “final”

### Author and institution

The items in `docs/dissertation/AUTHOR_INPUT_CHECKLIST.md` remain mandatory:

1. confirm the UDC and official title-page wording;
2. supply and verify both research-project numbers, titles, dates, and applicant role;
3. supply the target venue and status of article [A4];
4. supply six complete conference-publication records and verify DSMSI-2023;
5. confirm the official publication-count statement;
6. assign appendix letters and insert signed implementation-act scans;
7. enter final pagination after institutional typesetting.

There are nine `[AUTHOR: ...]` locations in the assembled manuscript. They are consolidated
administrative prompts, not missing scientific results.

### Venue and typesetting

- Select the [A4] venue and adapt length, sectioning, citation style, anonymization, and
  supplementary-material structure.
- Transfer equations, tables, captions, cross-references, and the live table of contents into the
  institutional dissertation template.
- Apply the required bibliographic standard (for example, DSTU 8302:2015 where mandated).
- Perform a new page-by-page render audit after the Markdown is transferred. The old 18 July
  render cannot certify the rewritten text.

### Distribution

- Publish artifact/checkpoint URLs with hashes, sizes, and licenses before claiming Tier-2 public
  reproduction.
- Add a CUDA lock or container for a genuinely public GPU reproduction path.
- Create a release tag only after the public archive and clean-clone reproduction are available.
- Treat `hvp.*` as the supported API. The frozen internal namespaces remain in the wheel for
  evidence compatibility and should not be advertised as stable.

### Scientific scope

The main result remains limited to the tested predictor, latent interface, budgets, and synthetic
instruments. Moving-MNIST is a meaningful external boundary but not natural-video validation. A
future natural-video disappearance/reappearance study, larger seed panels near the unresolved
frontier, and a direct checkpointing-versus-reversibility experiment inside the hierarchical
predictor would extend the dissertation; none is required to defend the present conditional
claim.

## Final editorial judgement

The article is now sharper than the earlier “final” version. It has a clear question, a memorable
but conditional answer, a valid main positive figure, explicit negative and mixed boundaries, and
enough mathematical detail to stand independently. Its greatest remaining risk is venue fit, not
story weakness.

The dissertation now has a coherent systems identity rather than four adjacent projects. The
two-memory decomposition makes the representation, reversible-computation, and hierarchical
prediction strands intellectually complementary without falsely claiming one end-to-end
pipeline. Its greatest remaining risks are institutional completion and the naturally limited
external validity of the fourth-article experiment, both stated openly.

Accordingly, the honest status is:

> **Scientific content final for the present evidence; submission packaging and public artifact
> distribution not final.**
