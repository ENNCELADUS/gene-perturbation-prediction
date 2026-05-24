"""Curated biological program scoring for perturbation deltas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CURATED_PROGRAM_GENE_SETS: dict[str, tuple[str, ...]] = {
    "cell_cycle_e2f": (
        "E2F1",
        "E2F2",
        "E2F3",
        "MYBL2",
        "MCM2",
        "MCM3",
        "MCM4",
        "MCM5",
        "MCM6",
        "MCM7",
        "PCNA",
        "TK1",
        "TYMS",
        "CDC6",
        "CDK1",
        "CCNE1",
        "CCNA2",
        "TOP2A",
        "RRM2",
        "BUB1",
        "BUB1B",
    ),
    "g2m_mitotic": (
        "CCNB1",
        "CCNB2",
        "CDC20",
        "PLK1",
        "AURKA",
        "AURKB",
        "CDK1",
        "BIRC5",
        "CENPF",
        "MKI67",
        "NUSAP1",
        "UBE2C",
        "KIF11",
        "KIF20A",
        "KIF2C",
    ),
    "myc_translation": (
        "MYC",
        "MAX",
        "NPM1",
        "NCL",
        "ODC1",
        "LDHA",
        "NME1",
        "EIF4E",
        "EIF4A1",
        "EIF3B",
        "RPL3",
        "RPL5",
        "RPLP0",
        "RPS3",
        "RPS6",
        "RPSA",
    ),
    "p53_apoptosis": (
        "TP53",
        "CDKN1A",
        "MDM2",
        "BBC3",
        "BAX",
        "BCL2L11",
        "PMAIP1",
        "GADD45A",
        "DDIT4",
        "FAS",
        "TNFRSF10B",
        "CASP3",
        "CASP7",
        "BIK",
    ),
    "dna_damage": (
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "BRCA1",
        "BRCA2",
        "RAD51",
        "FANCD2",
        "H2AFX",
        "DDB2",
        "XPC",
        "RPA1",
        "PARP1",
        "GADD45A",
    ),
    "upr_er_stress": (
        "ATF4",
        "ATF6",
        "DDIT3",
        "XBP1",
        "ERN1",
        "HSPA5",
        "HSP90B1",
        "DNAJB9",
        "HERPUD1",
        "PPP1R15A",
        "TRIB3",
        "ASNS",
    ),
    "interferon_inflammatory": (
        "STAT1",
        "STAT2",
        "IRF1",
        "IRF7",
        "ISG15",
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "MX1",
        "OAS1",
        "OAS2",
        "CXCL10",
        "DDX58",
    ),
    "ribosome_translation": (
        "RPL3",
        "RPL4",
        "RPL5",
        "RPL7",
        "RPL11",
        "RPL13A",
        "RPLP0",
        "RPS3",
        "RPS6",
        "RPS8",
        "RPS14",
        "RPS19",
        "EIF2S1",
        "EIF3A",
        "EIF4A1",
    ),
    "oxidative_phosphorylation": (
        "NDUFA1",
        "NDUFA2",
        "NDUFB5",
        "NDUFS1",
        "SDHA",
        "SDHB",
        "UQCRC1",
        "UQCRC2",
        "COX4I1",
        "COX5A",
        "ATP5F1A",
        "ATP5F1B",
        "ATP5MC1",
    ),
    "proteasome": (
        "PSMA1",
        "PSMA2",
        "PSMA3",
        "PSMA4",
        "PSMA5",
        "PSMB1",
        "PSMB2",
        "PSMB3",
        "PSMB4",
        "PSMB5",
        "PSMD1",
        "PSMD2",
        "PSMD4",
    ),
}


DEFAULT_PROGRAM_SCORE_SETS: tuple[str, ...] = tuple(CURATED_PROGRAM_GENE_SETS)


@dataclass(frozen=True)
class ProgramScoreResult:
    scores: pd.DataFrame
    score_columns: tuple[str, ...]
    qa_rows: list[dict[str, object]]


def build_program_scores(
    *,
    delta: np.ndarray,
    gene_symbols: list[str],
    program_sets: tuple[str, ...],
) -> ProgramScoreResult:
    """Compute signed mean delta scores for curated biological programs."""
    symbol_to_index = {symbol: index for index, symbol in enumerate(gene_symbols)}
    score_data: dict[str, np.ndarray] = {}
    qa_rows: list[dict[str, object]] = []

    for program_name in program_sets:
        genes = CURATED_PROGRAM_GENE_SETS.get(program_name)
        if genes is None:
            msg = f"Unknown program score set: {program_name}"
            raise ValueError(msg)
        matched_indices = [
            symbol_to_index[gene] for gene in genes if gene in symbol_to_index
        ]
        column_name = f"program_{program_name}_mean_delta"
        if matched_indices:
            with np.errstate(invalid="ignore"):
                score = np.nanmean(delta[:, matched_indices], axis=1)
            score = np.nan_to_num(score, nan=0.0)
        else:
            score = np.zeros(delta.shape[0], dtype=np.float32)
        score_data[column_name] = score.astype(np.float32)
        qa_rows.append(
            {
                "program": program_name,
                "n_genes": len(genes),
                "n_matched_expression_genes": len(matched_indices),
                "n_missing_expression_genes": len(genes) - len(matched_indices),
                "matched_fraction": len(matched_indices) / len(genes),
            }
        )

    scores = pd.DataFrame(score_data)
    return ProgramScoreResult(
        scores=scores,
        score_columns=tuple(scores.columns),
        qa_rows=qa_rows,
    )
