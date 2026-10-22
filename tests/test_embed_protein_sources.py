import numpy as np
import pytest

from src.data.build import embed_protein_sources as eps


def test_windows_follow_pie_esm2():
    assert eps.windows(1022) == [(0, 1022)]
    assert eps.windows(1200) == [(0, 1022), (511, 1200), (1022, 1200)]
    # PIE's loop keeps stepping to the end, so a tail window inside the last one stays.
    assert eps.windows(2100) == [
        (0, 1022),
        (511, 1533),
        (1022, 2044),
        (1533, 2100),
        (2044, 2100),
    ]
    with pytest.raises(ValueError):
        eps.windows(0)


def test_short_protein_is_the_plain_mean_of_its_residues():
    states = np.arange(10, dtype=np.float32).reshape(5, 2)
    pooled = eps.pool_residues(5, lambda start, end: states[start:end])
    np.testing.assert_allclose(pooled, states.mean(axis=0))


def test_long_protein_averages_each_residue_over_its_windows_first():
    length = 1200
    # A window's state for residue i is i plus the window's start: residues 511..1021
    # are in windows 0 and 511 (i + 255.5), residues from 1022 in windows 511 and 1022
    # (i + 766.5).
    pooled = eps.pool_residues(
        length,
        lambda start, end: (np.arange(start, end) + start)[:, None].astype(np.float32),
    )
    residue = np.arange(length, dtype=np.float64)
    expected = residue + np.where(
        residue < 511, 0.0, np.where(residue < 1022, 255.5, 766.5)
    )
    np.testing.assert_allclose(pooled, [expected.mean()], rtol=1e-6)


def test_residue_states_drop_bos_and_eos():
    hidden = np.arange(5 * 2, dtype=np.float32).reshape(1, 5, 2)
    np.testing.assert_array_equal(eps.residue_states(hidden, 3), hidden[0, 1:4])
    with pytest.raises(ValueError):
        eps.residue_states(hidden, 4)


def test_structure_is_used_only_when_its_sequence_equals_uniprot():
    members = frozenset(
        {
            eps.afdb_member("P1"),
            eps.afdb_member("P2"),
            eps.afdb_member("P3"),
            eps.afdb_member("P3", 2),
        }
    )
    assert eps.structure_status("P1", "MKV", members, "mkv") == eps.MATCHED
    assert eps.structure_status("P2", "MKV", members, "MKA") == "sequence_mismatch"
    assert eps.structure_status("P3", "MKV", members, None) == "fragmented"
    assert eps.structure_status("P4", "MKV", members, None) == "no_model"
    assert eps.afdb_member("P1") == "AF-P1-F1-model_v6.pdb.gz"


def test_resume_skips_finished_chunks(tmp_path):
    (tmp_path / "000000.npz").write_bytes(b"")
    (tmp_path / "000004.npz").write_bytes(b"")
    assert eps.pending_chunks(10, 2, tmp_path) == [2, 6, 8]


def test_steps_with_outputs_are_skipped(tmp_path, monkeypatch):
    for name in (eps.PROTEINS, "afdb.json", "esmc_600m.json", "esm3_open.json"):
        (tmp_path / name).write_text("{}")

    def refuse(*args, **kwargs):
        raise AssertionError("a finished step ran")

    for function in ("proteins", "afdb", "embed"):
        monkeypatch.setattr(eps, function, refuse)
    assert eps.main(["--steps", ",".join(eps.STEPS), "--out", str(tmp_path)]) == 0


def test_final_states_hook_equals_esmc_embeddings():
    pytest.importorskip("esm")
    torch = pytest.importorskip("torch")
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    from esm.tokenization import EsmSequenceTokenizer

    torch.manual_seed(0)
    model = ESMC(
        d_model=32,
        n_heads=2,
        n_layers=2,
        tokenizer=EsmSequenceTokenizer(),
        use_flash_attn=False,
    ).eval()
    hook = eps.FinalStates(model)
    sequence = "MKTAYIAKQRQISFVKSHFSRQ"
    output = model.logits(
        model.encode(ESMProtein(sequence=sequence)),
        LogitsConfig(sequence=True, return_embeddings=True),
    )
    np.testing.assert_allclose(
        hook.take(), output.embeddings.float().numpy(), rtol=1e-6
    )
    pooled = eps._embed(model, hook, sequence, None)
    np.testing.assert_allclose(
        pooled, output.embeddings[0, 1:-1].float().mean(0).numpy(), rtol=1e-5, atol=1e-6
    )
