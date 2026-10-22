import numpy as np

from src.data.embeddings import load_esm2_embeddings


def _write_npz(path):
    symbols = np.array(["TP53", "KRAS", "EGFR"], dtype=object)
    vectors = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    resolved = np.array([True, True, False])
    np.savez(path, symbols=symbols, vectors=vectors, resolved=resolved)


def test_load_drops_unresolved_genes(tmp_path):
    npz = tmp_path / "esm2.npz"
    _write_npz(npz)
    table = load_esm2_embeddings(npz)
    assert table.dim == 4
    assert set(table.vectors_by_symbol) == {"TP53", "KRAS"}
