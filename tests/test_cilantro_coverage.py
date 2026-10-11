import numpy as np
import pandas as pd

from src.experiments.cilantro_coverage import (
    bag_indices,
    context_rows,
    covered_rows,
    top_tokens,
    unversioned,
)


def test_top_tokens_ranks_by_value_over_median_and_skips_zeros():
    values = np.array([[4.0, 1.0, 0.0, 3.0], [0.0, 0.0, 5.0, 0.0]])
    medians = np.array([4.0, 0.1, 1.0, 1.0])
    present = top_tokens(values, medians, size=2)
    # Row 0 scores 1, 10, 0, 3: the median makes the smallest count the top token.
    assert present[0].tolist() == [False, True, False, True]
    # Fewer nonzero genes than the input size: all of them, never a zero gene.
    assert present[1].tolist() == [False, False, True, False]


def test_bag_indices_keeps_small_lines_whole_and_is_seeded():
    assert bag_indices(5, 128, 0).tolist() == [0, 1, 2, 3, 4]
    bag = bag_indices(1000, 128, 0)
    assert len(set(bag.tolist())) == 128 and np.all(np.diff(bag) > 0)
    assert np.array_equal(bag, bag_indices(1000, 128, 0))


def test_context_rows_count_rows_with_both_genes_covered():
    pairs = pd.DataFrame(
        {
            "context": ["K562"] * 3 + ["A549"],
            "model_id": ["ACH-1"] * 3 + ["ACH-2"],
            "split": ["train"] * 3 + ["validation"],
            "sl_label": [1, 0, 1, 1],
            "ensembl_a": ["E1", "E1", "E3", "E1"],
            "ensembl_b": ["E2", "E3", "E2", "E2"],
        }
    )
    assert covered_rows(pairs, {"E1", "E2"}).tolist() == [True, False, False, True]
    rows = {
        r["context"]: r
        for r in context_rows(pairs, {"ACH-1": {"E1", "E2"}, "ACH-2": set()}, "arm")
    }
    assert (rows["K562"]["rows"], rows["K562"]["positives"]) == (3, 2)
    assert rows["K562"]["covered_rows"] == 1
    assert rows["K562"]["covered_positives"] == 1
    assert rows["A549"]["covered_rows"] == 0


def test_unversioned_strips_ensembl_versions_only():
    assert unversioned("ENSG00000223972.5") == "ENSG00000223972"
    assert unversioned("ENSG00000223972") == "ENSG00000223972"
    assert unversioned("RP11-34P13.3") == "RP11-34P13.3"
