"""The direct readout's trunk (``pie.model.geneeffect``) and the arms that use it.

These need the vendored PIE (``.venv-pie``) and skip without it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pie")

import torch  # noqa: E402

from pie.model.geneeffect import (  # noqa: E402
    ChannelAttention,
    GeneEffectTrunk,
    chunked_attention,
    trunk_name_map,
    warm_start,
)
from src.model.stage_one import (  # noqa: E402
    DirectReadout,
    GeneEncoder,
    PairBatch,
)

TRUNK = {
    "d_model": 16,
    "n_latents": 4,
    "n_encoder_layers": 1,
    "n_processor_layers": 1,
    "n_decoder_layers": 1,
    "num_heads": 2,
    "ff_mult": 2,
    "dropout": 0.0,
    "attention_chunk": 1000,
    "warm_start": None,
}


def _trunk(chunk: int = 1000) -> GeneEffectTrunk:
    torch.manual_seed(0)
    return GeneEffectTrunk(
        d_model=16,
        n_latents=4,
        n_encoder_layers=2,
        n_processor_layers=1,
        n_decoder_layers=1,
        num_heads=2,
        ff_mult=2,
        dropout=0.0,
        channels={"tokens": None, "text": 5},
        attention_chunk=chunk,
    ).eval()


def test_chunked_attention_equals_one_pass():
    torch.manual_seed(0)
    q = torch.randn(3, 2, 4, 8)
    k, v = torch.randn(3, 2, 50, 8), torch.randn(3, 2, 50, 8)
    mask = torch.rand(3, 50) > 0.3
    mask[2] = False  # a line whose channel is empty reads zero
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=(mask | ~mask.any(1, keepdim=True))[:, None, None], scale=1.0
    )
    expected[2] = 0
    for chunk in (7, 50, 64):
        assert torch.allclose(
            chunked_attention(q, k, v, mask, chunk), expected, atol=1e-5
        )
    torch.manual_seed(1)
    whole = ChannelAttention(16, 2, chunk=1000).eval()
    pieces = ChannelAttention(16, 2, chunk=6).eval()
    pieces.load_state_dict(whole.state_dict())
    x, kv = torch.randn(2, 4, 16), torch.randn(2, 40, 16)
    keep = torch.rand(2, 40) > 0.2
    assert torch.allclose(whole(x, kv, keep), pieces(x, kv, keep), atol=1e-5)


def test_duplicating_a_channels_tokens_leaves_it_unchanged():
    trunk = _trunk()
    tokens, keep = torch.randn(2, 9, 16), torch.ones(2, 9, dtype=torch.bool)
    text, has_text = torch.randn(2, 1, 5), torch.tensor([[True], [False]])
    with torch.no_grad():
        once = trunk.encode({"tokens": (tokens, keep), "text": (text, has_text)})
        twice = trunk.encode(
            {
                "tokens": (torch.cat([tokens, tokens], 1), torch.cat([keep, keep], 1)),
                "text": (text, has_text),
            }
        )
    assert torch.allclose(once, twice, atol=1e-5)


def _direct(seed: int = 0) -> DirectReadout:
    torch.manual_seed(seed)
    return DirectReadout(
        gene_encoder=GeneEncoder({"gene_text": 6}, 16),
        text_dim=5,
        trunk=TRUNK,
        hidden=8,
    ).eval()


def _batch(genes: int = 3, lines: int = 2, seed: int = 0) -> PairBatch:
    generator = torch.Generator().manual_seed(seed)
    tokens = 7
    return PairBatch(
        gene_vectors={"gene_text": torch.randn(genes, 6, generator=generator)},
        gene_present={"gene_text": torch.ones(genes, dtype=torch.bool)},
        token_genes={"gene_text": torch.randn(tokens, 6, generator=generator)},
        token_gene_present={"gene_text": torch.ones(tokens, dtype=torch.bool)},
        token_values=torch.rand(lines, tokens, 4, generator=generator),
        token_present=torch.ones(lines, tokens, dtype=torch.bool),
        text=torch.randn(lines, 5, generator=generator),
        text_present=torch.tensor([True, False][:lines]),
        evidence=torch.rand(genes, lines, generator=generator),
        evidence_have=torch.ones(genes, lines, dtype=torch.bool),
    )


def _untie(model: torch.nn.Module) -> None:
    """Zero-initialised layers hide wrong wiring; give them weights."""
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.any():
                parameter.normal_(0.0, 0.3)


def test_the_untrained_direct_readout_predicts_the_base():
    z, v = _direct()(_batch())
    assert torch.equal(z, torch.zeros(3, 2)) and v.shape == (3, 2, 32)


def test_an_unlisted_gene_masks_only_its_own_evidence():
    model = _direct()
    _untie(model)
    batch = _batch()
    batch.evidence_have[1, 0] = False
    with torch.no_grad():
        before, _ = model(batch)
        batch.evidence[1, 0] = 50.0  # unlisted: its value is never read
        unchanged, _ = model(batch)
        batch.evidence[0, 0] = 50.0  # listed: moves gene 0 in line 0 only
        moved, _ = model(batch)
    assert torch.allclose(before, unchanged)
    changed = ~torch.isclose(moved, before)
    assert changed[0, 0] and changed.sum() == 1


def test_a_gene_never_trained_on_decodes():
    model = _direct()
    _untie(model)
    batch = _batch(genes=3)
    new = torch.randn(1, 6)
    batch.gene_vectors = {
        "gene_text": torch.cat([batch.gene_vectors["gene_text"][:1], new, new])
    }
    batch.evidence[2] = batch.evidence[1]
    with torch.no_grad():
        z, _ = model(batch)
    assert torch.isfinite(z).all() and torch.allclose(z[1], z[2])


def _pie_like_state(trunk: GeneEffectTrunk, d: int) -> dict[str, torch.Tensor]:
    """A released checkpoint's names for what the map reads, at our shapes."""
    names = trunk_name_map(
        trunk,
        {
            "n_latents": 4,
            "n_encoder_layers": 2,
            "n_processor_layers": 1,
            "n_decoder_layers": 1,
        },
    )
    ours = trunk.state_dict()
    state = {}
    for mine, theirs in names.items():
        for key, value in ours.items():
            if key == mine or key.startswith(mine + "."):
                state[theirs + key[len(mine) :]] = torch.full_like(value, 0.5)
    state["output_head.weight"] = torch.zeros(2, d)  # PIE's heads: not ours, never read
    return state


def test_warm_start_is_strict():
    trunk = _trunk()
    state = _pie_like_state(trunk, 16)
    names = trunk_name_map(
        trunk,
        {
            "n_latents": 4,
            "n_encoder_layers": 2,
            "n_processor_layers": 1,
            "n_decoder_layers": 1,
        },
    )
    fresh = warm_start(trunk, state, names)
    assert torch.all(trunk.encoder[0].channels["text"].attn.q_proj.weight == 0.5)
    assert torch.all(trunk.latents == 0.5)
    assert any(name.startswith("encoder.0.channels.tokens.") for name in fresh)
    assert not any(name.startswith("encoder.0.channels.text.") for name in fresh)
    missing = dict(state)
    missing.pop("encoder.0.attn.q_proj.weight")
    with pytest.raises(ValueError, match="tensors differ"):
        warm_start(_trunk(), missing, names)
    wrong = dict(state)
    wrong["gene_norm.weight"] = torch.ones(17)
    with pytest.raises(ValueError, match="shape"):
        warm_start(_trunk(), wrong, names)
    with pytest.raises(ValueError, match="encoder blocks"):
        trunk_name_map(
            trunk,
            {
                "n_latents": 4,
                "n_encoder_layers": 1,
                "n_processor_layers": 1,
                "n_decoder_layers": 1,
            },
        )
