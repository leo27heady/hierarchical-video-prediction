"""Bind dissertation ``[REF: ...]`` markers and assemble a numbered bibliography.

The chapter sources retain stable descriptive markers so future edits do not create numbering
churn. This script resolves those markers against the verified-key master bibliography, numbers
external sources by first appearance, and writes assembly-ready Markdown copies. It fails closed
on an unknown descriptor or a missing master entry.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "dissertation" / "rewrite"
OUTPUT = ROOT / "docs" / "dissertation" / "final"
MASTER = SOURCE / "07_REFERENCES.md"

CHAPTERS = [
    "01_VSTUP.md",
    "02_ROZDIL1.md",
    "03_ROZDIL2.md",
    "04_ROZDIL3.md",
    "05_ROZDIL4.md",
    "06_VYSNOVKY.md",
]


# Ordered rules intentionally resolve the narrowest descriptors first. Values are stable keys
# from 07_REFERENCES.md. Several broad review claims appropriately cite more than one source.
RULES: list[tuple[str, tuple[str, ...]]] = [
    (r"^uncertainty decomposition$", ("Kendall2017uncertainty",)),
    (r"^calibration of modern neural networks$", ("Guo2017calibration",)),
    (r"^medical image reconstruction hallucination risk$", ("Bhadra2021hallucinations",)),
    (
        r"^physics-informed and uncertainty-aware remote-sensing imputation$",
        ("Ebel2023uncrtaints", "Huot2022wildfire", "Shaddy2024wildfire"),
    ),
    (
        r"^clip-score validity and limitations$",
        ("Hessel2021clipscore", "Otani2023t2ieval", "Thrush2022winoground"),
    ),
    (
        r"^uncertainty-aware resource allocation$",
        ("Kendall2017uncertainty", "Guo2017calibration"),
    ),
    (r"^revolve checkpoint scheduling$", ("Griewank2000",)),
    (r"^chen et al\.? 2016$", ("Chen2016ckpt",)),
    (r"^scheduled sampling$", ("Bengio2015",)),
    (
        r"^sequence-level training / self-forcing$",
        ("Ranzato2016", "Huang2025selfforcing"),
    ),
    (
        r"recent long-memory retrieval and benchmark work",
        (
            "WorldPack2025",
            "Peng2026car",
            "Yu2026memlearner",
            "Wei2026gimworld",
            "Zhang2026mbench",
            "Chen2026memobench",
            "LSSVWM2025",
        ),
    ),
    (
        r"efficient long-sequence state and attention alternatives",
        (
            "Dai2019txl",
            "Rae2020compressive",
            "Wu2022memorizing",
            "Kitaev2020reformer",
            "Beltagy2020longformer",
            "Gu2023mamba",
        ),
    ),
    (
        r"modern video diffusion and token generation",
        (
            "Ho2022video",
            "Blattmann2023videoldm",
            "Blattmann2023svd",
            "Singer2023makeavideo",
            "Ho2022imagenvideo",
            "Villegas2022phenaki",
            "Yu2023magvit",
            "Yu2024magvit2",
        ),
    ),
    (
        r"interactive and autonomous world models",
        (
            "Bruce2024genie",
            "Hu2023gaia1",
            "Ha2018worldmodels",
            "Hafner2020dreamer",
            "Hafner2023dreamerv3",
            "Micheli2023iris",
        ),
    ),
    (
        r"hierarchical latent representation foundations",
        ("Razavi2019vqvae2", "Child2021vdvae", "Vahdat2020nvae", "Sonderby2016laddervae"),
    ),
    (
        r"systems foundations for complexity, information, state, and control",
        ("Simon1962", "Shannon1948", "Kalman1960", "Ashby1956"),
    ),
    (
        r"bootstrap, clustered resampling, and simultaneous inference",
        ("Efron1979", "Cameron2008cluster", "WestfallYoung1993", "RomanoWolf2005"),
    ),
    (
        r"ssim, lpips, fid, and fvd",
        ("Wang2004ssim", "Zhang2018lpips", "Heusel2017", "Unterthiner2019fvd"),
    ),
    (r"attention scaling and memory-efficient exact attention", ("Vaswani2017", "Dao2022flash")),
    (r"hungarian assignment", ("Kuhn1955",)),
    (r"moving-mnist original", ("Srivastava2015",)),
    (r"taxibj", ("Zhang2017taxibj",)),
    (r"svg.*/.*savp", ("Denton2018", "Lee2018savp")),
    (r"harvey.*fdm|\bfdm\b", ("Harvey2022",)),
    (r"survey of large-scale generative video", ("SoraSurvey2024",)),
    (r"black-box critique|interpretability of deep generative", ("Lipton2018", "Rudin2019")),
    (r"ddpm|ho et al", ("Ho2020",)),
    (r"classical systems theory", ("Bertalanffy1968",)),
    (r"\bgan\b|goodfellow et al", ("Goodfellow2014",)),
    (r"\bvae\b.*kingma|kingma.*welling", ("Kingma2014",)),
    (r"systems-under-risk|system methodology monograph", ("Zgurovsky2007", "Zgurovsky2007en")),
    (r"hierarchical systems theory", ("Mesarovic1970",)),
    (r"encoder.generator.decoder", ("BondTaylor2022",)),
    (r"ldm/vae latent-space", ("Rombach2022", "Kingma2014")),
    (r"critique of flat latent", ("Saxena2021", "Zakharov2023dlh", "Yan2023teco")),
    (r"hallucination phenomenology", ("Ji2023",)),
    (r"\bddim\b", ("Song2021ddim",)),
    (r"\bldm\b", ("Rombach2022",)),
    (r"\bdit\b|peebles.*xie", ("Peebles2023",)),
    (r"sora technical", ("Sora2024",)),
    (
        r"long-term spatial-memory|memory-augmented world-model|geometry-grounded long-video",
        ("Pasukonis2022", "LSSVWM2025", "VidMem2025", "WorldPack2025"),
    ),
    (r"exposure bias", ("Ranzato2016", "Bengio2015")),
    (r"dosovitskiy.*vit|^vit$", ("Dosovitskiy2021",)),
    (r"arnab.*vivit|^vivit$", ("Arnab2021",)),
    (r"video swin|liu.*swin", ("Liu2022swin",)),
    (r"\bblt\b|pagnoni", ("Pagnoni2024",)),
    (r"vq-vae|van den oord", ("Oord2017",)),
    (r"videogpt", ("Yan2021videogpt",)),
    (r"memory maze.*teco-minecraft", ("Pasukonis2022", "Yan2023teco")),
    (r"\bvta\b|variational temporal abstraction", ("Kim2019vta",)),
    (r"\bteco\b|long-video discrete-token", ("Yan2023teco",)),
    (r"vq optimization|codebook collapse", ("Oord2017", "Yan2021videogpt")),
    (r"clockwork vae|cw-vae", ("Saxena2021",)),
    (r"millivid", ("MilliVid",)),
    (r"clockwork rnn|koutn", ("Koutnik2014",)),
    (r"statistical-physics foundations|boltzmann-distribution", ("LeCun2006", "Ackley1985")),
    (r"score matching", ("HyvSM2005", "Song2019score")),
    (r"energy-based transformers|ebt.*gladstone", ("Gladstone2025ebt",)),
    (r"ebm training-cost", ("Du2019", "Nijkamp2019")),
    (r"gradient checkpointing", ("Chen2016ckpt", "Griewank2000")),
    (r"neural ode", ("Chen2018ode",)),
    (r"revnet", ("Gomez2017",)),
    (r"revvit", ("Mangalam2022",)),
    (r"stability of complex systems", ("Khalil2002",)),
    (r"\bfid\b", ("Heusel2017",)),
    (r"image-to-image diffusion", ("Saharia2022",)),
    (r"dreambooth", ("Ruiz2023",)),
    (r"\blora\b", ("Hu2022lora",)),
    (r"controlnet", ("Zhang2023controlnet",)),
    (r"stylealigned", ("Hertz2024",)),
    (r"multi-stage visual representation", ("Marr1982",)),
    (r"^clip$", ("Radford2021",)),
    (r"information leakage", ("Vaswani2017",)),
    (r"cross-entropy classification", ("Goodfellow2016book",)),
    (r"absolute positional encoding", ("Vaswani2017",)),
    (r"relative positional bias", ("Shaw2018", "Liu2022swin")),
    (r"\brope\b|rotary embedding", ("Su2024rope",)),
    (r"gpu memory as the binding", ("Rajbhandari2020", "Korthikanti2022")),
    (r"werbos.*bptt", ("Werbos1990",)),
    (r"truncated bptt", ("Williams1990",)),
    (r"mangalam.*gomez|gomez.*mangalam", ("Gomez2017", "Mangalam2022")),
    (r"mackay.*reversible rnn", ("MacKay2018",)),
    (r"behrmann.*i-resnet|numerical invertibility", ("Behrmann2019", "Behrmann2021")),
    (r"maclaurin.*reversible", ("Maclaurin2015",)),
    (r"doodl|edict|mali", ("Wallace2023doodl", "Wallace2023edict", "Zhuang2021mali")),
    (r"alignprop|draft", ("Prabhudesai2023", "Clark2024draft")),
    (r"improved ddpm|nichol.*dhariwal", ("Nichol2021",)),
    (r"progressive distillation|salimans.*ho", ("Salimans2022",)),
    (r"arxiv 2505\.20171", ("LSSVWM2025",)),
    (r"arxiv 2512\.02473", ("WorldPack2025",)),
    (r"arxiv 2506\.05284", ("VidMem2025",)),
    (r"openstl", ("Tan2023openstl",)),
    (r"svg.*denton", ("Denton2018",)),
    (r"savp.*lee", ("Lee2018savp",)),
    (r"\bdiamond\b", ("Alonso2024",)),
    (r"mean-field entropy-collapse", ("MeanField2024",)),
    (r"ebt-policy", ("EBTPolicy2025",)),
    (r"self-forcing", ("Huang2025selfforcing",)),
]


def _normalize(text: str) -> str:
    return " ".join(text.replace("–", "-").replace("—", "-").split()).strip()


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def load_master() -> dict[str, str]:
    text = MASTER.read_text(encoding="utf-8")
    entries: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"- \*\*([^*]+)\*\* — (.*?)(?:\s+✓(?:\s.*)?)?$", line)
        if match:
            entries[match.group(1)] = match.group(2).strip()
    return entries


def resolve_descriptor(descriptor: str) -> list[str]:
    descriptor = _normalize(descriptor)
    keys: list[str] = []
    # Nested REF forms occur inside one Markdown marker, for example
    # ``[REF: SVG ...; REF: SAVP ...]``.
    parts = re.split(r";\s*REF:\s*", descriptor, flags=re.IGNORECASE)
    for part in parts:
        norm = part.casefold()
        for pattern, candidates in RULES:
            if re.search(pattern, norm, flags=re.IGNORECASE):
                keys.extend(candidates)
                break
        else:
            raise ValueError(f"Unresolved reference descriptor: {part!r}")
    return _unique(keys)


def bind_text(text: str, numbers: dict[str, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        keys = resolve_descriptor(match.group(1))
        for key in keys:
            if key not in numbers:
                numbers[key] = len(numbers) + 1
        return "[" + ", ".join(str(numbers[key]) for key in keys) + "]"

    return re.sub(r"\[REF:\s*(.*?)\]", replace, text, flags=re.DOTALL)


def main() -> None:
    entries = load_master()
    numbers: dict[str, int] = {}
    bound: dict[str, str] = {}
    for name in CHAPTERS:
        bound[name] = bind_text((SOURCE / name).read_text(encoding="utf-8"), numbers)

    missing = [key for key in numbers if key not in entries]
    if missing:
        raise KeyError(f"Missing master bibliography entries: {missing}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, text in bound.items():
        (OUTPUT / name).write_text(text, encoding="utf-8", newline="\n")

    ordered = sorted(numbers, key=numbers.get)
    lines = [
        "# References",
        "",
        "External sources are numbered by first appearance in the assembled dissertation.",
        "",
    ]
    for key in ordered:
        lines.append(f"{numbers[key]}. {entries[key]}")
    lines.append("")
    (OUTPUT / "07_REFERENCES_BOUND.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )

    unresolved = []
    for name, text in bound.items():
        if re.search(r"\[REF(?::|\])", text):
            unresolved.append(name)
    if unresolved:
        raise RuntimeError(f"Unbound reference markers remain in: {unresolved}")
    print(f"BOUND_REFERENCES={len(numbers)}")
    print(f"OUTPUT={OUTPUT}")


if __name__ == "__main__":
    main()
