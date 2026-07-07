# 07 — СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ (master bibliography + binding table)

> **Status (2026-07-07):** the master reference list for BOTH texts (dissertation СПИСОК
> ВИКОРИСТАНИХ ДЖЕРЕЛ and the article bibliography — the article cites a subset of the same
> keys). Built from the full [REF:] inventory of `docs/dissertation/rewrite/*` +
> `docs/draft/CH*` (104 unique descriptors, 120 occurrences, extracted 2026-07-07).
>
> **Numbering policy:** entries carry STABLE KEYS, not numbers. Final numbers are assigned
> mechanically at assembly (E8) in order of first mention, after (a) the author verifies flagged
> entries and (b) the author's own publications are bound. This avoids renumbering churn.
> **Formatting:** entries below are in a neutral scholarly format; the UA pass reformats the
> dissertation copy to **ДСТУ 8302:2015** (Latin-script sources remain Latin). The article uses
> the venue's style.
> **Verification legend:** ✓ = high-confidence bibliographic data; **[VERIFY]** = check against
> the PDF/publisher page before submission (standing debt, COMPLETION_PLAN E6); **[AUTHOR]** =
> only the author can supply. Per the campaign's honesty rule, EVERY entry gets a PDF-check at
> the E6 verification pass — ✓ marks confidence, not completed verification.

---

## A. Публікації здобувача (the applicant's own works — bind first at assembly)

| key | entry | status |
|---|---|---|
| KulykA1 | Kulyk L. R. [Hierarchy/BLT-adaptation article: entropy-based adaptive spatio-temporal decomposition, V-BLT two-stage hierarchical video prediction, procedural data synthesis]. UA professional journal (spec. 124). | **[AUTHOR: exact entry — venue, year, vol., pages]** |
| KulykA2 | Kulyk L. Temporal Modeling with Reversible Transformers. *Neural Computing and Applications* (Springer), 2025. Scopus-indexed, sole author. | **[AUTHOR: vol./pages/DOI; confirm Q1 or soften]** |
| KulykA3 | Kulyk L. R. [Consistency-criteria article: the criteria apparatus, conceptual consistency of generative models]. UA professional journal (spec. 124). | **[AUTHOR: exact entry]** |
| KulykA4 | Kulyk L. R. [The fourth article — hierarchical latent video model with coarse-level long memory; memory-horizon testbed; matched-compute mechanism study; reversibility]. In finalization. | **[AUTHOR: target venue; status by defense]** |
| KulykT1–T6 | Six conference theses (VNTU L–LIII conференції; «Проблеми вищої математичної освіти»; DSMSI-2023). | **[AUTHOR: identify all six; DSMSI-2023 unverified]** |

## B. Master list (grouped; key → full entry → status)

### B1. Diffusion and iterative generation

- **Ho2020** — Ho J., Jain A., Abbeel P. Denoising Diffusion Probabilistic Models. *NeurIPS 33*, 2020. ✓
- **Song2021ddim** — Song J., Meng C., Ermon S. Denoising Diffusion Implicit Models. *ICLR*, 2021. ✓
- **Nichol2021** — Nichol A., Dhariwal P. Improved Denoising Diffusion Probabilistic Models. *ICML*, 2021. ✓
- **Rombach2022** — Rombach R., Blattmann A., Lorenz D., Esser P., Ommer B. High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR*, 2022. ✓
- **Peebles2023** — Peebles W., Xie S. Scalable Diffusion Models with Transformers (DiT). *ICCV*, 2023. ✓
- **Salimans2022** — Salimans T., Ho J. Progressive Distillation for Fast Sampling of Diffusion Models. *ICLR*, 2022. ✓
- **Saharia2022** — Saharia C., Chan W., Chang H., et al. Palette: Image-to-Image Diffusion Models. *SIGGRAPH*, 2022. ✓
- **Sora2024** — OpenAI (Brooks T., Peebles B., et al.). Video Generation Models as World Simulators (Sora technical report), 2024. ✓
- **HyvSM2005** — Hyvärinen A. Estimation of Non-Normalized Statistical Models by Score Matching. *JMLR* 6, 2005. ✓
- **Song2019score** — Song Y., Ermon S. Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS 32*, 2019. ✓

### B2. Foundational generative models and representation learning

- **Kingma2014** — Kingma D. P., Welling M. Auto-Encoding Variational Bayes. *ICLR*, 2014. ✓
- **Goodfellow2014** — Goodfellow I., Pouget-Abadie J., Mirza M., et al. Generative Adversarial Networks. *NeurIPS 27*, 2014. ✓
- **Oord2017** — van den Oord A., Vinyals O., Kavukcuoglu K. Neural Discrete Representation Learning (VQ-VAE). *NeurIPS 30*, 2017. ✓
- **Heusel2017** — Heusel M., Ramsauer H., Unterthiner T., et al. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID). *NeurIPS 30*, 2017. ✓
- **Radford2021** — Radford A., Kim J. W., Hallacy C., et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). *ICML*, 2021. ✓
- **BondTaylor2022** — Bond-Taylor S., Leach A., Long Y., Willcocks C. G. Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models. *IEEE TPAMI* 44(11), 2022. ✓
- **Goodfellow2016book** — Goodfellow I., Bengio Y., Courville A. *Deep Learning*. MIT Press, 2016. ✓

### B3. Transformers, attention, positional encoding, tokenization

- **Vaswani2017** — Vaswani A., Shazeer N., Parmar N., et al. Attention Is All You Need. *NeurIPS 30*, 2017. ✓
- **Dosovitskiy2021** — Dosovitskiy A., Beyer L., Kolesnikov A., et al. An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT). *ICLR*, 2021. ✓
- **Arnab2021** — Arnab A., Dehghani M., Heigold G., et al. ViViT: A Video Vision Transformer. *ICCV*, 2021. ✓
- **Liu2022swin** — Liu Z., Ning J., Cao Y., et al. Video Swin Transformer. *CVPR*, 2022. ✓
- **Shaw2018** — Shaw P., Uszkoreit J., Vaswani A. Self-Attention with Relative Position Representations. *NAACL*, 2018. ✓
- **Su2024rope** — Su J., Ahmed M., Lu Y., et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing* 568, 2024 (arXiv:2104.09864, 2021). ✓
- **Pagnoni2024** — Pagnoni A., Pasunuru R., Rodriguez P., et al. Byte Latent Transformer: Patches Scale Better Than Tokens. arXiv:2412.09871, 2024. ✓
- **Yan2021videogpt** — Yan W., Zhang Y., Abbeel P., Srinivas A. VideoGPT: Video Generation using VQ-VAE and Transformers. arXiv:2104.10157, 2021. ✓

### B4. Hierarchical / long-horizon video prediction and world-model memory

- **Saxena2021** — Saxena V., Ba J., Hafner D. Clockwork Variational Autoencoders. *NeurIPS 34*, 2021. ✓
- **Zakharov2023dlh** — Zakharov A., Guo Q., Fountas Z. Long-Horizon Video Prediction Using a Dynamic Latent Hierarchy. arXiv:2212.14376, 2023. **[VERIFY venue/year]**
- **Yan2023teco** — Yan W., Hafner D., James S., Abbeel P. Temporally Consistent Transformers for Video Generation (TECO). *ICML*, 2023. **[VERIFY author list]**
- **VillarCorrales2022** — Villar-Corrales A., Karapetyan A., Boltres A., Behnke S. MSPred: Video Prediction at Multiple Spatio-Temporal Scales. *BMVC*, 2022. **[VERIFY]**
- **Yin2023nuwa** — Yin S., Wu C., Yang H., et al. NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation. *ACL*, 2023. ✓
- **MilliVid** — [MIT + TRI hierarchical latent video diffusion — the architecture family the Chapter 3 system miniaturizes]. **[VERIFY: exact citation from the sibling `../millivid` repo README/paper]**
- **Harvey2022** — Harvey W., Naderiparizi S., Masrani V., Weilbach C., Wood F. Flexible Diffusion Modeling of Long Videos (FDM). *NeurIPS 35*, 2022. ✓
- **Pasukonis2022** — Pasukonis J., Lillicrap T., Hafner D. Evaluating Long-Term Memory in 3D Mazes (Memory Maze). arXiv:2210.13383, 2022. ✓
- **LSSVWM2025** — [Long-term spatial-memory / state-space video world model]. arXiv:2505.20171, 2025. **[VERIFY title/authors]**
- **WorldPack2025** — [Memory-packing for video world models]. arXiv:2512.02473, 2025. **[VERIFY title/authors]**
- **VidMem2025** — [Video world models with long-term spatial memory]. arXiv:2506.05284, 2025. **[VERIFY title/authors]**
- **Denton2018** — Denton E., Fergus R. Stochastic Video Generation with a Learned Prior (SVG). *ICML*, 2018. ✓
- **Lee2018savp** — Lee A. X., Zhang R., Ebert F., et al. Stochastic Adversarial Video Prediction (SAVP). arXiv:1804.01523, 2018. ✓
- **Alonso2024** — Alonso E., Jelley A., Micheli V., et al. Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND). *NeurIPS 37*, 2024. ✓
- **Huang2025selfforcing** — Huang X., et al. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion. *NeurIPS*, 2025. **[VERIFY authors/arXiv id]**

### B5. Energy-based models

- **LeCun2006** — LeCun Y., Chopra S., Hadsell R., Ranzato M., Huang F. A Tutorial on Energy-Based Learning. In *Predicting Structured Data*, MIT Press, 2006. ✓
- **Ackley1985** — Ackley D. H., Hinton G. E., Sejnowski T. J. A Learning Algorithm for Boltzmann Machines. *Cognitive Science* 9(1), 1985. ✓
- **Du2019** — Du Y., Mordatch I. Implicit Generation and Modeling with Energy-Based Models. *NeurIPS 32*, 2019. ✓
- **Nijkamp2019** — Nijkamp E., Hill M., Zhu S.-C., Wu Y. N. On Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model. *NeurIPS 32*, 2019. **[VERIFY-fit as the training-cost citation]**
- **Gladstone2025ebt** — Gladstone A., Alonso [et al.]. Energy-Based Transformers [are Scalable Learners and Thinkers]. arXiv:2507.02092, 2025. **[VERIFY exact author list/title]**
- **EBTPolicy2025** — [Energy-Based Transformer policies for robotic control]. arXiv:2510.27545, 2025. **[VERIFY title/authors]**
- **MeanField2024** — [Mean-field analysis of gradient-descent sampling entropy collapse]. arXiv:2403.08362, 2024. **[VERIFY title/authors]**

### B6. Reversible computation, memory-efficient training

- **Gomez2017** — Gomez A. N., Ren M., Urtasun R., Grosse R. B. The Reversible Residual Network: Backpropagation Without Storing Activations. *NeurIPS 30*, 2017. ✓
- **Mangalam2022** — Mangalam K., Fan H., Li Y., et al. Reversible Vision Transformers. *CVPR*, 2022. ✓
- **MacKay2018** — MacKay M., Vicol P., Ba J., Grosse R. B. Reversible Recurrent Neural Networks. *NeurIPS 31*, 2018. ✓
- **Maclaurin2015** — Maclaurin D., Duvenaud D., Adams R. P. Gradient-Based Hyperparameter Optimization Through Reversible Learning. *ICML*, 2015. ✓
- **Chen2016ckpt** — Chen T., Xu B., Zhang C., Guestrin C. Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174, 2016. ✓
- **Griewank2000** — Griewank A., Walther A. Algorithm 799: revolve — An Implementation of Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation. *ACM TOMS* 26(1), 2000. ✓
- **Chen2018ode** — Chen R. T. Q., Rubanova Y., Bettencourt J., Duvenaud D. Neural Ordinary Differential Equations. *NeurIPS 31*, 2018. ✓
- **Behrmann2019** — Behrmann J., Grathwohl W., Chen R. T. Q., Duvenaud D., Jacobsen J.-H. Invertible Residual Networks (i-ResNet). *ICML*, 2019. ✓
- **Behrmann2021** — Behrmann J., Vicol P., Wang K.-C., Grosse R., Jacobsen J.-H. Understanding and Mitigating Exploding Inverses in Invertible Neural Networks. *AISTATS*, 2021. ✓
- **Werbos1990** — Werbos P. J. Backpropagation Through Time: What It Does and How to Do It. *Proceedings of the IEEE* 78(10), 1990. ✓
- **Williams1990** — Williams R. J., Peng J. An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network Trajectories (truncated BPTT). *Neural Computation* 2(4), 1990. ✓
- **Wallace2023edict** — Wallace B., Gokul A., Naik N. EDICT: Exact Diffusion Inversion via Coupled Transformations. *CVPR*, 2023. ✓
- **Wallace2023doodl** — Wallace B., Gokul A., Ermon S., Naik N. End-to-End Diffusion Latent Optimization Improves Classifier Guidance (DOODL). *ICCV*, 2023. ✓
- **Zhuang2021mali** — Zhuang J., Dvornek N. C., Tatikonda S., Duncan J. S. MALI: A Memory-Efficient and Reverse-Accurate Integrator for Neural ODEs. *ICLR*, 2021. ✓
- **Prabhudesai2023** — Prabhudesai M., Goyal A., Pathak D., Fragkiadaki K. Aligning Text-to-Image Diffusion Models with Reward Backpropagation (AlignProp). arXiv:2310.03739, 2023. ✓
- **Clark2024draft** — Clark K., Vicol P., Swersky K., Fleet D. J. Directly Fine-Tuning Diffusion Models on Differentiable Rewards (DRaFT). *ICLR*, 2024. ✓
- **Rajbhandari2020** — Rajbhandari S., Rasley J., Ruwase O., He Y. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. *SC20*, 2020. ✓ *(GPU memory as binding constraint)*
- **Korthikanti2022** — Korthikanti V., Casper J., Lym S., et al. Reducing Activation Recomputation in Large Transformer Models. arXiv:2205.05198, 2022. **[VERIFY-fit]**

### B7. Video-prediction benchmarks and baselines (Chapter 4–5 comparisons)

- **Shi2015** — Shi X., Chen Z., Wang H., et al. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. *NeurIPS 28*, 2015. ✓
- **Wang2017predrnn** — Wang Y., Long M., Wang J., Gao Z., Yu P. S. PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs. *NeurIPS 30*, 2017. ✓
- **Wang2018predrnnpp** — Wang Y., Gao Z., Long M., Wang J., Yu P. S. PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning. *ICML*, 2018. ✓
- **Gao2022simvp** — Gao Z., Tan C., Wu L., Li S. Z. SimVP: Simpler yet Better Video Prediction. *CVPR*, 2022. ✓
- **Tan2023tau** — Tan C., Gao Z., Wu L., et al. Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning (TAU). *CVPR*, 2023. ✓
- **Tan2023openstl** — Tan C., Li S., Gao Z., et al. OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning. *NeurIPS 36 (D&B)*, 2023. ✓
- **Srivastava2015** — Srivastava N., Mansimov E., Salakhutdinov R. Unsupervised Learning of Video Representations using LSTMs (Moving MNIST). *ICML*, 2015. ✓
- **Schuldt2004** — Schuldt C., Laptev I., Caputo B. Recognizing Human Actions: A Local SVM Approach (KTH). *ICPR*, 2004. ✓
- **Zhang2017taxibj** — Zhang J., Zheng Y., Qi D. Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction (TaxiBJ). *AAAI*, 2017. ✓
- **Ranzato2016** — Ranzato M., Chopra S., Auli M., Zaremba W. Sequence Level Training with Recurrent Neural Networks (exposure bias). *ICLR*, 2016. ✓
- **Bengio2015** — Bengio S., Vinyals O., Jaitly N., Shazeer N. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. *NeurIPS 28*, 2015. ✓

### B8. Controllability, consistency, evaluation of generative systems (Chapter 1 criteria)

- **Zhang2023controlnet** — Zhang L., Rao A., Agrawala M. Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet). *ICCV*, 2023. ✓
- **Ruiz2023** — Ruiz N., Li Y., Jampani V., et al. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. *CVPR*, 2023. ✓
- **Hu2022lora** — Hu E. J., Shen Y., Wallis P., et al. LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*, 2022. ✓
- **Hertz2024** — Hertz A., Voynov A., Fruchter S., Cohen-Or D. Style Aligned Image Generation via Shared Attention. *CVPR*, 2024. ✓
- **Ji2023** — Ji Z., Lee N., Frieske R., et al. Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys* 55(12), 2023. ✓ *(hallucination phenomenology)*
- **Lipton2018** — Lipton Z. C. The Mythos of Model Interpretability. *Communications of the ACM* 61(10), 2018. ✓ *(black-box critique)*
- **Rudin2019** — Rudin C. Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead. *Nature Machine Intelligence* 1, 2019. ✓
- **SoraSurvey2024** — Liu Y., Zhang K., Li Y., et al. Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. arXiv:2402.17177, 2024. **[VERIFY-fit as the large-scale-video survey]**
- **Marr1982** — Marr D. *Vision: A Computational Investigation into the Human Representation and Processing of Visual Information*. W. H. Freeman, 1982. **[VERIFY-fit for "two-stage visual perception"]**

### B9. Systems analysis, hierarchical systems, stability (the 124 spine — Chapter 1)

- **Zgurovsky2007** — Згуровський М. З., Панкратова Н. Д. *Основи системного аналізу*. Київ: Видавнича група BHV, 2007. **[VERIFY edition/pages; supervisor may prefer a specific edition]**
- **Zgurovsky2007en** — Zgurovsky M. Z., Pankratova N. D. *System Analysis: Theory and Applications*. Springer, 2007. ✓
- **Mesarovic1970** — Mesarović M. D., Macko D., Takahara Y. *Theory of Hierarchical, Multilevel Systems*. Academic Press, 1970. ✓ *(hierarchical systems theory)*
- **Bertalanffy1968** — von Bertalanffy L. *General System Theory: Foundations, Development, Applications*. George Braziller, 1968. ✓ *(classical systems theory)*
- **Khalil2002** — Khalil H. K. *Nonlinear Systems* (3rd ed.). Prentice Hall, 2002. ✓ *(stability of complex systems)*
- **UA-риз** — [«Системи під ризиком» methodology — supervisor-preferred UA source(s) on системний аналіз в умовах невизначеності та ризику.] **[AUTHOR: Prof. Mokin's preferred source(s); 2–3 UA entries expected here for the council]**

## C. Binding table (inline [REF: descriptor] → key; substitution at assembly)

| descriptor (as it appears inline) | key(s) |
|---|---|
| A1 / A2 (article self-references) | KulykA1 / KulykA2 |
| absolute positional encoding, Vaswani et al. 2017 | Vaswani2017 |
| AlignProp 2023; DRaFT 2023 | Prabhudesai2023; Clark2024draft |
| Arnab et al. 2021, ViViT / ViViT | Arnab2021 |
| arXiv 2505.20171 / 2512.02473 / 2506.05284 (+ LSSVWM/WorldPack/VidMem forms) | LSSVWM2025; WorldPack2025; VidMem2025 |
| Behrmann et al. (all forms; i-ResNet line; numerical invertibility) | Behrmann2019; Behrmann2021 |
| black-box critique / interpretability | Lipton2018; Rudin2019 |
| BLT / Pagnoni et al. 2024, BLT | Pagnoni2024 |
| Chen et al. 2016 (+ Griewank & Walther, treeverse) / gradient checkpointing | Chen2016ckpt; Griewank2000 |
| Chen et al. 2018, Neural ODE / Neural ODE | Chen2018ode |
| classical systems theory textbook | Bertalanffy1968 |
| CLIP | Radford2021 |
| Clockwork VAE / CW-VAE | Saxena2021 |
| ControlNet | Zhang2023controlnet |
| critique of flat latent spaces | Saxena2021; Zakharov2023dlh; Yan2023teco |
| cross-entropy classification objective | Goodfellow2016book |
| DDIM (all forms) | Song2021ddim |
| DDPM (all forms) | Ho2020 |
| DIAMOND (all forms) | Alonso2024 |
| DiT (all forms) | Peebles2023 |
| DOODL / EDICT / MALI (all forms) | Wallace2023doodl; Wallace2023edict; Zhuang2021mali |
| Dosovitskiy et al. 2021, ViT / ViT | Dosovitskiy2021 |
| DreamBooth | Ruiz2023 |
| EBM training-cost analyses | Du2019; Nijkamp2019 |
| EBT (all forms, 2507.02092, Gladstone) | Gladstone2025ebt |
| EBT-Policy (both forms) | EBTPolicy2025 |
| encoder–generator–decoder architecture survey | BondTaylor2022 |
| Energy-Based Transformers | Gladstone2025ebt |
| exposure bias in sequence generation | Ranzato2016; Bengio2015 |
| FID | Heusel2017 |
| GAN (both forms) | Goodfellow2014 |
| Gomez et al. 2017 / RevNet (+ Mangalam, RevViT forms) | Gomez2017; Mangalam2022 |
| GPU memory as the binding constraint | Rajbhandari2020; Korthikanti2022 |
| hallucination phenomenology | Ji2023 |
| Harvey et al. 2022 / FDM | Harvey2022 |
| hierarchical systems theory | Mesarovic1970 |
| image-to-image diffusion | Saharia2022 |
| improved DDPM — Nichol & Dhariwal, 2021 | Nichol2021 |
| information leakage in causal sequence modeling | Vaswani2017 *(causal masking; +[AUTHOR-CHOICE if a dedicated source is wanted)* |
| LDM (both forms) | Rombach2022 |
| Liu et al. 2022, Video Swin / Video Swin | Liu2022swin |
| long-term spatial-memory benchmarks; memory-augmented world-model evaluation; geometry-grounded probes | Pasukonis2022; LSSVWM2025; VidMem2025; WorldPack2025 |
| LoRA | Hu2022lora |
| MacKay et al. 2018 (both forms) | MacKay2018 |
| Maclaurin et al. 2015 (both forms) | Maclaurin2015 |
| mean-field entropy-collapse, 2403.08362 | MeanField2024 |
| MilliVid (all forms; hierarchical latent video diffusion) | MilliVid |
| OpenSTL | Tan2023openstl |
| progressive distillation — Salimans & Ho | Salimans2022 |
| relative positional bias, Shaw et al. 2018 / Swin | Shaw2018; Liu2022swin |
| RoPE (all forms; RoPE-ND / multi-axis) | Su2024rope *(RoPE-ND itself = KulykA1)* |
| score matching | HyvSM2005; Song2019score |
| Self-Forcing (both forms) | Huang2025selfforcing |
| Sora technical report | Sora2024 |
| stability of complex systems | Khalil2002 |
| statistical-physics foundations of EBM; Boltzmann-distribution modeling | LeCun2006; Ackley1985 |
| StyleAligned | Hertz2024 |
| survey of large-scale generative video models | SoraSurvey2024 |
| SVG / SAVP (all forms) | Denton2018; Lee2018savp |
| system methodology monograph; systems-under-risk methodology | Zgurovsky2007; Zgurovsky2007en; UA-риз |
| TECO (all forms) | Yan2023teco |
| truncated BPTT | Williams1990 |
| two-stage visual perception | Marr1982 |
| VAE, Kingma & Welling | Kingma2014 |
| van den Oord et al. 2017, VQ-VAE (both forms) | Oord2017 |
| VideoGPT | Yan2021videogpt |
| Werbos 1990, BPTT | Werbos1990 |
| NUWA-XL / MSPred (article Ch. 7 forms) | Yin2023nuwa; VillarCorrales2022 |
| Memory Maze / TECO-Minecraft (article Ch. 7) | Pasukonis2022; Yan2023teco |

## D. Assembly instructions (E8)

1. Author fills section A + the three **[AUTHOR]** items in B/C (UA-риз sources; MilliVid entry;
   information-leakage choice) and clears every **[VERIFY]** flag against the PDFs (E6 debt).
2. Script pass replaces each inline `[REF: descriptor]` with `[N]` (or `[N, M]`) by the binding
   table, numbering by order of first mention: 01_VSTUP → 02 → 03 → 04 → 05 → 06 (the article
   numbers independently by its venue style from the same keys).
3. The dissertation copy is reformatted to ДСТУ 8302:2015 at the UA pass; the count line in
   VSTUP («Список джерел містить N найменувань») is filled from the final count
   (current census: ~85 external entries + ~10 author entries; the old draft's target of ~145
   is met once the Chapter-1 topical sections are bound with multi-source citations —
   [AUTHOR: decide whether to enrich §1.1.2 applications and §1.3 with additional per-topic
   sources at the verification pass; the current list covers every inline claim]).
