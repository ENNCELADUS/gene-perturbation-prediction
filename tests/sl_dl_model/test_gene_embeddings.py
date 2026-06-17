import numpy as np

from sl_dl_model.gene_embeddings import (
    align_esm2_to_universe,
    load_esm2_embeddings,
)


def _write_npz(path):
    symbols = np.array(["TP53", "KRAS", "EGFR"], dtype=object)
    vectors = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    resolved = np.array([True, True, False])
    np.savez(path, symbols=symbols, vectors=vectors, resolved=resolved)


def test_load_and_align(tmp_path):
    npz = tmp_path / "esm2.npz"
    _write_npz(npz)
    table = load_esm2_embeddings(npz)
    assert table.dim == 4
    assert set(table.vectors_by_symbol) == {"TP53", "KRAS"}  # unresolved dropped

    symbols = np.array(["KRAS", "TP53", "UNKNOWN"], dtype=object)
    emb, mask = align_esm2_to_universe(table, symbols, "zero")
    assert emb.shape == (3, 4)
    assert mask.tolist() == [1, 1, 0]
    assert np.allclose(emb[2], 0.0)  # fallback for uncovered


def test_align_global_mean_fallback(tmp_path):
    npz = tmp_path / "esm2.npz"
    _write_npz(npz)
    table = load_esm2_embeddings(npz)
    symbols = np.array(["TP53", "ZZZ"], dtype=object)
    emb, mask = align_esm2_to_universe(table, symbols, "global_mean")
    assert mask.tolist() == [1, 0]
    assert not np.allclose(emb[1], 0.0)  # global-mean fallback non-zero
