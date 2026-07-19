# Front Matter

---

## 1. Title Page

- **Topic (UA, approved, verbatim — fixed):** «Системний аналіз та інтелектуальні технології
  консистентного моделювання процесів ієрархічної генерації відеоданих»
- **Topic (EN):** *Systems Analysis and Intelligent Technologies for Consistent Modeling of
  Hierarchical Video Data Generation Processes*
- Candidate: Kulyk Leonid Ruslanovych. Specialty 124 «System Analysis», field 12 «Information
  Technologies». Supervisor: Prof. O. B. Mokin, Dr. Sc. (Eng.). VNTU, Vinnytsia — 2026.
- UDC: 004.054:[004.032.26+004.85] **[AUTHOR: confirm the UDC classification and the title-page
  wording against the approved institutional record.]**

---

## 2. Abstract

Kulyk L. R. *Systems Analysis and Intelligent Technologies for Consistent Modeling of
Hierarchical Video Data Generation Processes*. — Qualifying scientific work submitted as a
manuscript.

The dissertation addresses the scientific and technical problem of modelling long multi-step
video processes under bounded computational resources. A video generator must retain
memory-relevant history, preserve task-relevant visual detail, and support differentiation through
long temporal computations. Treating these requirements as one undifferentiated “memory” problem
obscures which resource is limiting and which intervention is appropriate.

The work develops a systems methodology that separates total training and inference memory into
parameter, optimizer, context/state, activation, and transient components:

$$
M_{\mathrm{system}}=
M_{\mathrm{parameters}}+
M_{\mathrm{optimizer}}+
M_{\mathrm{context/state}}+
M_{\mathrm{activations}}+
M_{\mathrm{transient}}.
$$

Two complementary resource axes are investigated. Hierarchical or multirate context packaging acts
on \(M_{\mathrm{context/state}}\) by retaining distant history at reduced spatial resolution or
temporal density. Reversible temporal computation acts on \(M_{\mathrm{activations}}\) by
reconstructing intermediate states during reverse replay. Their savings were measured separately
and are not assumed to multiply automatically.

The dissertation develops methods for entropy-guided adaptive decomposition of video data,
N-dimensional rotary positional encoding, reversible optimization of long temporal processes, and
multi-stream latent video generation. The final hierarchical predictor uses a frozen two-level
continuous latent representation, a shared joint-attention backbone, v-parameterized window
diffusion, and autoregressive rollout. A purpose-built panning-world instrument provides exact
geometric observability when content leaves and re-enters the viewport. Context packages are
compared at explicit context-token budgets, and inference is based on paired clip-cluster
resampling with simultaneous max-statistic intervals for registered comparison families.

On a hash-pinned untouched holdout, hierarchy is confirmed where a token-matched flat
full-resolution window cannot reach the relevant re-entry past. The registered 14-to-48-frame
reach-band effects are +0.0980 and +0.0914 across two training seeds. The simultaneous tested
frontier confirms hierarchy at 896 and 1,152 context tokens, leaves 1,792 unresolved, and confirms
flat full-resolution context at 2,432 and 3,072. In the frozen interface, 48 fine frames require
3,072 context tokens, while 48 coarse frames plus a two- or six-frame fine tail require 896 or
1,152. Thus the confirmed hierarchical packages provide the same nominal 48-frame addressable
reach with 2.67–3.43 times fewer context tokens; this is a reach-efficiency result, not a claim of
equal information or equal prediction error.

The principal directions persist across registered sampler allocations and an independently
trained codec, although the upper inversion is less precise under alternate sampler or codec
conditions. A preregistered long-rollout study confirms an early hierarchy benefit at three
training seeds but fails to replicate a seed-0 late inversion; no stable trust boundary is
claimed. A protected persistent-object 48-to-300 Moving-MNIST study provides a mixed external
boundary: hierarchy better preserves spatial support and trajectory continuity, whereas flat
context better preserves digit identity. Both systems remain difficult in absolute terms.

The reversible-computation strand independently demonstrates exact reverse replay and constant
activation-memory scaling with temporal horizon in the tested recurrence, while identifying
gradient checkpointing as the appropriate practical competitor and delimiting cases in which
reversibility does not pay. The dissertation therefore contributes an integrated systems
methodology and research software toolkit, not one experimentally validated end-to-end model that
combines every strand.

The practical design rule is conditional: compress long-lived state when distant reach is the
binding constraint; preserve dense detail when fidelity is limiting; reconstruct activations when
the training tape dominates; and evaluate storage, addressing, and self-fed reuse as distinct
failure modes.

**Keywords:** systems analysis; hierarchical video generation; long-horizon video prediction;
context packaging; memory horizon; temporal consistency; reversible computation; activation
memory; diffusion transformer; autoregressive prediction; geometric observability; clip-cluster
inference.

---

## 3. Applicant's Publications

The following stable labels are used throughout the dissertation:

- **[A1]** — Kulyk L. R., Mokin O. B. “Scaling Video Prediction with Spatio-Temporal Patches,”
  *Visnyk of Vinnytsia Politechnical Institute*, 2025, No. 5, pp. 129–139,
  DOI 10.31649/1997-9266-2025-182-5-129-139.
- **[A2]** — Kulyk L. “Temporal modeling with reversible transformers” (TempVerseFormer),
  *Neural Computing and Applications*, 2026, Vol. 38, article 30,
  DOI 10.1007/s00521-025-11707-3, sole author, Scopus-indexed.
- **[A3]** — Kulyk L. R., Mokin O. B. “Methods for Ensuring Consistent Generation in Diffusion
  Models,” *Visnyk of Vinnytsia Politechnical Institute*, 2024, No. 4, pp. 75–85,
  DOI 10.31649/1997-9266-2024-175-4-75-85.
- **[A4]** — the fourth article, *When Does Hierarchical Video Memory Pay? Reach–Fidelity
  Trade-offs under Fixed Context Budgets*, in finalization.
  **[AUTHOR: supply target venue and status at submission/defence.]**
- Six conference theses.
  **[AUTHOR: supply the six verified bibliographic entries.]**

Publication-administration facts that cannot be established from the scientific repository are
collected in `docs/dissertation/AUTHOR_INPUT_CHECKLIST.md`; they must not be inferred.

---

## 4. Symbols and Abbreviations

| Abbreviation | Expansion |
|---|---|
| VNTU | Vinnytsia National Technical University |
| AE | Autoencoder |
| AR | Autoregressive |
| BPTT | Backpropagation Through Time |
| CI | Confidence Interval |
| CRN | Common Random Numbers |
| DDIM | Denoising Diffusion Implicit Models |
| DDPM | Denoising Diffusion Probabilistic Models |
| DiT | Diffusion Transformer |
| EBM | Energy-Based Model |
| EBT | Energy-Based Transformer |
| EMA | Exponential Moving Average |
| FID | Fréchet Inception Distance |
| FVD | Fréchet Video Distance |
| fg-MSE | Foreground Mean Squared Error |
| GPU | Graphics Processing Unit |
| LPIPS | Learned Perceptual Image Patch Similarity |
| MCMC | Markov Chain Monte Carlo |
| MSE | Mean Squared Error |
| NFE | Number of Function Evaluations |
| OOM | Out of Memory |
| RoPE-ND | N-dimensional Rotary Positional Embedding |
| SSIM | Structural Similarity Index Measure |
| V-BLT | Video Byte-Latent Transformer |
| VAE | Variational Autoencoder |

---

## 5. Table of Contents

The submission document must generate this list as a live table of contents with page numbers and
cross-references:

1. Introduction.
2. Chapter 1. Systems Analysis of Hierarchical Video Generation and Its Consistency Criteria.
3. Chapter 2. Mathematical Methods for Uncertainty Estimation and Resource-Efficient Modelling of
   Multi-Step Video Processes.
4. Chapter 3. A Hierarchical Mathematical Model for Video Generation with Long Multi-Stream
   Context.
5. Chapter 4. Experimental Evaluation and Practical Application.
6. General Conclusions.
7. References.
8. Appendices: applicant publications; implementation acts; registered experimental
   configurations; supplementary evidence tables; and reproduction information.
