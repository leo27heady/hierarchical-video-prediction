"""Create provenance records for pre-existing author-generated dissertation figures.

The restructured dissertation reuses several repository figures that predate the
new figure-generation pass.  This script records those files without modifying
their pixels.  It computes dimensions, byte sizes, and SHA-256 digests directly
from the linked PNG assets so the ledger can be verified mechanically.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "restructured" / "existing_figure_provenance.json"


FIGURES = [
    {
        "figure_id": "1.5",
        "file": "docs/figures/fig31_consistency_landscape.png",
        "generator": "scripts/make_dissertation_foundation_figures.py",
        "data_source": "conceptual consistency criteria synthesized in Chapter 1",
        "role": "dissertation-native conceptual landscape",
    },
    {
        "figure_id": "1.23",
        "file": "docs/figures/fig35_fixed_grid_tokenization.png",
        "generator": "scripts/make_dissertation_foundation_figures.py",
        "data_source": "fixed-grid spatial-allocation analysis in Chapter 1",
        "role": "dissertation-native conceptual tokenization schematic",
    },
    {
        "figure_id": "1.24",
        "file": "docs/figures/fig39_flat_long_context_barrier.png",
        "generator": "scripts/make_dissertation_foundation_figures.py",
        "data_source": "retained-context reach formulation in Chapters 1 and 3",
        "role": "dissertation-native conceptual reach schematic",
    },
    {
        "figure_id": "2.15",
        "file": "docs/figures/fig47_rope_nd.png",
        "generator": "scripts/make_dissertation_foundation_figures.py",
        "data_source": "verified RoPE-ND mathematical contract and implementation configuration",
        "role": "dissertation-native implementation-informed schematic",
    },
    {
        "figure_id": "1.15",
        "file": "docs/figures/restructured/published_2024_fig4_diffusion_pipeline.png",
        "generator": "scripts/make_restructured_dissertation_figures.py",
        "data_source": "official 2024 article PDF, original Figure 4",
        "role": "published-reused-unchanged literature-review illustration",
        "published": True,
    },
    {
        "figure_id": "1.16",
        "file": "docs/figures/restructured/published_2024_fig6_img2img.png",
        "generator": "scripts/make_restructured_dissertation_figures.py",
        "data_source": "official 2024 article PDF, original Figure 6",
        "role": "published-reused-unchanged literature-review illustration",
        "published": True,
    },
    {
        "figure_id": "1.18",
        "file": "docs/figures/restructured/published_2024_fig8_inversion.png",
        "generator": "scripts/make_restructured_dissertation_figures.py",
        "data_source": "official 2024 article PDF, original Figure 8",
        "role": "published-reused-unchanged literature-review illustration",
        "published": True,
    },
    {
        "figure_id": "3.2",
        "file": "docs/figures/fig40_context_package_comparison.png",
        "generator": "scripts/make_dissertation_foundation_figures.py",
        "data_source": "frozen package registry and exact context-token arithmetic",
        "role": "dissertation-native configuration schematic",
    },
    {
        "figure_id": "3.5",
        "file": "docs/figures/fig22_token_sequence_layout.png",
        "generator": "scripts/make_dissertation_missing_figures.py",
        "data_source": "implemented tokenization path and frozen principal package configuration",
        "role": "dissertation-native implementation schematic",
    },
    {
        "figure_id": "3.9",
        "file": "docs/figures/fig16_context_packaging_architecture.png",
        "generator": "scripts/make_rft_core_schematics.py",
        "data_source": "experiments/configs/e2_untouched_confirmation.json and implemented predictor interface",
        "role": "dissertation-native implementation schematic",
    },
    {
        "figure_id": "3.11",
        "file": "docs/figures/fig23_rollout_ring_buffer.png",
        "generator": "scripts/make_dissertation_missing_figures.py",
        "data_source": "implemented rollout recurrence and frozen 48-coarse/6-fine package",
        "role": "dissertation-native implementation schematic",
    },
    {
        "figure_id": "3.14",
        "file": "docs/figures/fig15_loopmom_observability_schematic.png",
        "generator": "scripts/make_rft_core_schematics.py",
        "data_source": "experiments/configs/e2_untouched_confirmation.json and LoopMom generator geometry",
        "role": "dissertation-native mechanism-identification schematic",
    },
    {
        "figure_id": "4.1",
        "file": "docs/figures/fig63_executed_configuration_matrix.png",
        "generator": "scripts/make_dissertation_foundation_figures.py",
        "data_source": "complete configuration/evidence registry and terminal experiment reports",
        "role": "dissertation-native evidence-coverage matrix",
    },
    {
        "figure_id": "4.2",
        "file": "docs/figures/fig24_generation_objectives.png",
        "generator": "scripts/make_dissertation_missing_figures.py",
        "data_source": "implemented regression, diffusion, and energy-head interfaces",
        "role": "dissertation-native mechanism-comparison schematic",
    },
    {
        "figure_id": "4.3",
        "file": "docs/figures/fig29_mechanism_single_step_samples_compact.png",
        "generator": "scripts/make_dissertation_compact_galleries.py",
        "data_source": "fixed first-three-record crop of docs/figures/fig29_mechanism_single_step_samples.png",
        "role": "artifact-derived fixed-index qualitative comparison",
    },
    {
        "figure_id": "4.5",
        "file": "docs/figures/fig28_memory_gate_optimization.png",
        "generator": "scripts/make_dissertation_missing_figures.py",
        "data_source": (
            "logs_rhcp/mem_gate_full_s0.log, logs_rhcp/mem_gate_full_s1.log, "
            "logs_rhcp/mem_gate_full_s0_t20k.log, and logs_rhcp/mem_gate_full_s1_t20k.log"
        ),
        "role": "artifact-derived development diagnostic",
    },
    {
        "figure_id": "4.7",
        "file": "docs/figures/article_confirmed_frontier.png",
        "generator": "scripts/make_rft_result_figures.py",
        "data_source": "artifacts/confirmation_e2/results.json",
        "role": "artifact-derived untouched simultaneous frontier",
    },
    {
        "figure_id": "4.8",
        "file": "docs/figures/fig7_samples_claim1_compact.png",
        "generator": "scripts/make_dissertation_compact_galleries.py",
        "data_source": (
            "fixed clip-0 crop of docs/figures/fig7_samples_claim1.png; seed 1 and horizons "
            "1, 24, 60, 100, and 148"
        ),
        "role": "artifact-derived fixed-index compact descriptive gallery",
    },
    {
        "figure_id": "4.9",
        "file": "docs/figures/fig17_head_package_seed_heterogeneity.png",
        "generator": "scripts/make_rft_result_figures.py",
        "data_source": "artifacts/e4b_head_package/results.json",
        "role": "artifact-derived five-seed factorial result",
    },
    {
        "figure_id": "4.10",
        "file": "docs/figures/fig19_sampler_nfe_robustness.png",
        "generator": "scripts/make_rft_result_figures.py",
        "data_source": "artifacts/e7c_sampler_nfe/results.json",
        "role": "artifact-derived sampler and NFE robustness result",
    },
    {
        "figure_id": "4.11",
        "file": "docs/figures/fig20_alternate_codec_robustness.png",
        "generator": "scripts/make_rft_result_figures.py",
        "data_source": "artifacts/e7a_codec_robustness/results.json",
        "role": "artifact-derived independently trained codec robustness result",
    },
    {
        "figure_id": "4.12",
        "file": "docs/figures/fig18_trust_negative_replication.png",
        "generator": "scripts/make_rft_result_figures.py",
        "data_source": "artifacts/e5a_trust_replication/results.json",
        "role": "artifact-derived registered negative replication",
    },
    {
        "figure_id": "4.13",
        "file": "docs/figures/fig21_pose_indexed_raw_memory.png",
        "generator": "scripts/make_rft_result_figures.py",
        "data_source": "artifacts/e8a_pose_mosaic/results.json",
        "role": "artifact-derived privileged explicit-access reference",
    },
    {
        "figure_id": "4.14",
        "file": "docs/figures/fig22_phasec_v2_boundary.png",
        "generator": "scripts/make_rft_result_figures.py",
        "data_source": "artifacts/phasec_v2/results.json",
        "role": "artifact-derived persistent-object support/identity boundary",
    },
    {
        "figure_id": "E.1",
        "file": "docs/figures/fig29_mechanism_single_step_samples.png",
        "generator": "scripts/make_dissertation_missing_figures.py",
        "data_source": "logs_rhcp/step1_mech_clean_reg.png and logs_rhcp/step1_mech_clean_diff.png",
        "role": "artifact-derived full qualitative comparison",
    },
    {
        "figure_id": "E.2",
        "file": "docs/figures/fig7_samples_claim1.png",
        "generator": "scripts/make_samples.py",
        "data_source": (
            "frozen LoopMom train/test arrays, hier_ae_mom.pt, and seed-1 hier-long/flat-budget "
            "30,000-step checkpoints; fixed clip indices 0, 1, 3, and 5 and horizons "
            "1, 24, 60, 100, and 148"
        ),
        "role": "artifact-derived fixed-index full descriptive gallery",
    },
]


# Files below are the concrete inputs used to interpret or render each asset.
# Conceptual schematics point to the chapter that defines the construct; empirical
# plots point to their frozen result files.  The lists are deliberately separate
# from ``data_source`` so their existence can be checked mechanically.
SOURCE_FILES = {
    "1.5": ["docs/dissertation/restructured_english/02_CHAPTER_1.md"],
    "1.23": ["docs/dissertation/condensed_translated_humanized_ukrainian/02_CHAPTER_1.md"],
    "1.24": ["docs/dissertation/condensed_translated_humanized_ukrainian/02_CHAPTER_1.md"],
    "2.15": ["docs/dissertation/condensed_translated_humanized_ukrainian/03_CHAPTER_2.md"],
    "1.15": ["docs/figures/restructured/sources/Kulyk_Mokin_2024_official.pdf"],
    "1.16": ["docs/figures/restructured/sources/Kulyk_Mokin_2024_official.pdf"],
    "1.18": ["docs/figures/restructured/sources/Kulyk_Mokin_2024_official.pdf"],
    "3.2": ["experiments/configs/e2_untouched_confirmation.json", "hvp_rev/rhvd.py"],
    "3.5": ["experiments/configs/e2_untouched_confirmation.json", "hvp_rev/rhvd.py"],
    "3.9": ["experiments/configs/e2_untouched_confirmation.json", "hvp_rev/rhvd.py"],
    "3.11": ["experiments/configs/e2_untouched_confirmation.json", "hvp_rev/rhvd.py"],
    "3.14": [
        "experiments/configs/e2_untouched_confirmation.json",
        "docs/DATA_CARD_LOOPMOM.md",
    ],
    "4.1": [
        "docs/dissertation/rewrite/appendices/APPENDIX_CH4_CONFIGURATION_AND_EVIDENCE_REGISTRY.md",
        "docs/FINAL_PROGRAMME_REPORT_2026-07-18.md",
    ],
    "4.2": ["hvp_rev/rhvd.py", "hvp_rev/step1_mech.py", "hvp_rev/ebt_reference_control.py"],
    "4.3": ["docs/figures/fig29_mechanism_single_step_samples.png"],
    "4.5": [
        "logs_rhcp/mem_gate_full_s0.log",
        "logs_rhcp/mem_gate_full_s1.log",
        "logs_rhcp/mem_gate_full_s0_t20k.log",
        "logs_rhcp/mem_gate_full_s1_t20k.log",
    ],
    "4.7": ["artifacts/confirmation_e2/results.json"],
    "4.8": ["docs/figures/fig7_samples_claim1.png"],
    "4.9": ["artifacts/e4b_head_package/results.json"],
    "4.10": ["artifacts/e7c_sampler_nfe/results.json"],
    "4.11": ["artifacts/e7a_codec_robustness/results.json"],
    "4.12": ["artifacts/e5a_trust_replication/results.json"],
    "4.13": ["artifacts/e8a_pose_mosaic/results.json"],
    "4.14": ["artifacts/phasec_v2/results.json"],
    "E.1": ["logs_rhcp/step1_mech_clean_reg.png", "logs_rhcp/step1_mech_clean_diff.png"],
    "E.2": [
        "docs/DATA_CARD_LOOPMOM.md",
        "data/loopmom_train_3000x100.npy",
        "data/loopmom_train_3000x100_actions.npy",
        "data/loopmom_test_500x200.npy",
        "data/loopmom_test_500x200_actions.npy",
        "logs_rhcp/hier_ae_mom.pt",
        "logs_rhcp/rhvd_hier-long_s1_t30000.pt",
        "logs_rhcp/rhvd_flat-budget_s1_t30000.pt",
    ],
}


QUALITATIVE_LOOPMOM_FIGURES = {"4.3", "4.8", "E.1", "E.2"}
LAYOUT_NOTES = {
    "4.1": (
        "Wide nine-column matrix: place on a landscape or dedicated full-width page. "
        "At ordinary portrait text width its 7.3-pt source labels fall below the blueprint's "
        "8.5-pt preferred minimum. Table 4.2 remains the authoritative readable registry."
    ),
    "E.2": (
        "Tall fixed-index gallery: reserve a dedicated portrait page and do not reduce it "
        "below full printable height; review labels in the final typeset proof."
    ),
}


CHAPTER_FILES = {
    "1": ROOT / "docs" / "dissertation" / "condensed_translated_humanized_ukrainian" / "02_CHAPTER_1.md",
    "2": ROOT / "docs" / "dissertation" / "condensed_translated_humanized_ukrainian" / "03_CHAPTER_2.md",
    "3": ROOT / "docs" / "dissertation" / "condensed_translated_humanized_ukrainian" / "04_CHAPTER_3.md",
    "4": ROOT / "docs" / "dissertation" / "condensed_translated_humanized_ukrainian" / "05_CHAPTER_4.md",
    "E": ROOT / "docs" / "dissertation" / "condensed_translated_humanized_ukrainian" / "08_APPENDICES.md",
}


def png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def main() -> None:
    records = []
    for specification in FIGURES:
        path = ROOT / specification["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        generator = ROOT / specification["generator"]
        if not generator.is_file():
            raise FileNotFoundError(generator)

        source_inventory = []
        for source_file in SOURCE_FILES[specification["figure_id"]]:
            source_path = ROOT / source_file
            available = source_path.is_file()
            source_inventory.append(
                {
                    "file": source_file,
                    "available": available,
                    "bytes": source_path.stat().st_size if available else None,
                }
            )

        chapter = CHAPTER_FILES[specification["figure_id"].split(".", 1)[0]]
        chapter_text = chapter.read_text(encoding="utf-8")
        relative_link = "../../" + specification["file"].removeprefix("docs/")
        link_count = len(re.findall(rf"\]\({re.escape(relative_link)}\)", chapter_text))
        if link_count != 1:
            raise ValueError(
                f"expected exactly one live manuscript link to {relative_link} in {chapter}; "
                f"found {link_count}"
            )

        width, height = png_dimensions(path)
        if specification.get("published"):
            rights_status = (
                "Author-created artwork reused from Kulyk and Mokin (2024), published under "
                "CC BY 4.0; authors retain copyright."
            )
            embedded_third_party_pixels = False
            permission_status = "cleared under the article's CC BY 4.0 licence"
        elif specification["figure_id"] in QUALITATIVE_LOOPMOM_FIGURES:
            rights_status = (
                "Author-generated composite from project-produced predictions and ground-truth "
                "frames; no publisher artwork was copied. The author-created composite uses "
                "MNIST-derived glyphs solely as dataset content, with Moving-MNIST provenance "
                "and the exact local data manifest recorded separately."
            )
            embedded_third_party_pixels = True
            permission_status = "cleared for scholarly dissertation use; dataset provenance and citation recorded"
        else:
            rights_status = (
                "Author-generated repository artwork; no third-party or publisher figure pixels "
                "were identified in this asset. Upstream dataset and result-artifact distribution "
                "terms remain separate from copyright in the plotted artwork."
            )
            embedded_third_party_pixels = False
            permission_status = "no external artwork permission required"
        if specification.get("published"):
            asset_class = "published-reused-unchanged"
        elif specification["role"].startswith("artifact-derived"):
            asset_class = "author-generated-artifact-derived"
        elif "implementation" in specification["role"] or "configuration" in specification["role"]:
            asset_class = "author-generated-implementation-informed"
        elif "evidence" in specification["role"]:
            asset_class = "author-generated-evidence-synthesis"
        else:
            asset_class = "author-generated-conceptual"
        records.append(
            {
                "schema_version": "1.0",
                **specification,
                "asset_class": asset_class,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "pixel_dimensions": [width, height],
                "source": (
                    "Kulyk and Mokin, Methods for Ensuring Consistent Generation in Diffusion Models (2024)"
                    if specification.get("published")
                    else "author-generated repository figure"
                ),
                "source_paper": (
                    "Kulyk and Mokin, Visnyk VNTU 175(4), 2024, DOI 10.31649/1997-9266-2024-175-4-75-85"
                    if specification.get("published")
                    else None
                ),
                "original_figure": specification["figure_id"],
                "source_page": None,
                "extraction_method": None,
                "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
                "data_or_concept_source": specification["data_source"],
                "source_files": source_inventory,
                "live_manuscript_file": chapter.relative_to(ROOT).as_posix(),
                "live_manuscript_link_count": link_count,
                "modification_status": (
                    "published appearance reused unchanged"
                    if specification.get("published")
                    else "deterministic fixed crop from a frozen full montage"
                    if specification["generator"].endswith("make_dissertation_compact_galleries.py")
                    else "pre-existing PNG reused without pixel modification"
                ),
                "embedded_third_party_pixels": embedded_third_party_pixels,
                "publication_reuse_status": (
                    "published author-owned artwork reused under CC BY 4.0"
                    if specification.get("published")
                    else "author-created composite or repository artwork; no published figure pixels copied"
                ),
                "rights_status": rights_status,
                "permission_status": permission_status,
                "caption_attribution": (
                    f"{'Appendix' if specification['figure_id'].startswith('E.') else 'Chapter'} "
                    f"{specification['figure_id'].split('.', 1)[0]} Figure "
                    f"{specification['figure_id']}; caption includes an explicit author/source classification."
                ),
                "layout_note": LAYOUT_NOTES.get(specification["figure_id"]),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
