"""The gene-knowledge, cell-set model: the untrained model predicts the gene mean, the
pair score does not change with the order of the two genes, the order and the number
of the cells do not change the output, and a line without cells reads its mean token."""

from __future__ import annotations

import pytest
import torch

from src.model.sl_embedding import PAIR_WIDTH, WIDTH, GeneLineModel, pair_in_line

GENES, EXPRESSION, CELL = 12, 6, 8


NO_PAIR = torch.zeros(4, 0)  # the line head of ``model()`` reads no pair-in-line


def model(
    seed: int = 0, line_input: str = "model", pair_width: int = 0
) -> GeneLineModel:
    torch.manual_seed(seed)
    present = torch.ones(GENES, dtype=torch.bool)
    present[3] = False
    built = GeneLineModel(
        {"text": (torch.randn(GENES, 5), present)},
        expression_dim=EXPRESSION,
        cell_dim=CELL,
        hidden=16,
        head_hidden=16,
        gene_in_line={"own": 3},
        line_input=line_input,
        pair_in_line=pair_width,
    )
    # a trained state: the zero layers and the deviation are no longer zero
    for parameter in (built.mean_token[-1].weight, built.cell_token[-1].weight,
                      built.deviation):  # fmt: skip
        torch.nn.init.normal_(parameter, std=0.5)
    return built


def inputs(lines: int = 3, cells: int = 7):
    return (
        torch.randn(lines, EXPRESSION),
        torch.randn(lines, cells, CELL),
        torch.ones(lines, cells, dtype=torch.bool),
    )


def test_untrained_model_predicts_the_gene_mean():
    torch.manual_seed(0)
    fresh = GeneLineModel(
        {"text": (torch.randn(GENES, 5), torch.ones(GENES, dtype=torch.bool))},
        expression_dim=EXPRESSION, cell_dim=CELL, hidden=16, head_hidden=16,
        gene_in_line={}, line_input="model", pair_in_line=0,
    )  # fmt: skip
    expression, cells, mask = inputs()
    mean, tokens = fresh.tokens(expression, cells)
    z = fresh.dependency(fresh.gene_vectors()[:-1], mean, tokens, mask)
    assert z.shape == (3, GENES) and torch.all(z == 0)


def test_gene_vectors_end_with_a_zero_row_and_backbone_leaves_the_head_out():
    built = model()
    u = built.gene_vectors()
    assert u.shape == (GENES + 1, WIDTH) and torch.all(u[-1] == 0)
    head = {id(p) for p in built.head.parameters()}
    assert head and not head & {id(p) for p in built.backbone()}
    assert len(built.backbone()) + len(head) + len(built.line_parameters()) == len(
        list(built.parameters())
    )


def test_cell_order_and_cell_count_do_not_change_the_embedding():
    built = model()
    expression, cells, mask = inputs()
    u = built.gene_vectors()[:-1]
    mean, tokens = built.tokens(expression, cells)
    base = built.embed(u, mean, tokens, mask)
    order = torch.randperm(cells.shape[1])
    assert torch.allclose(base, built.embed(u, mean, tokens[:, order], mask), atol=1e-5)
    twice = torch.cat([tokens, tokens], dim=1)
    both = torch.cat([mask, mask], dim=1)
    assert torch.allclose(base, built.embed(u, mean, twice, both), atol=1e-5)


def test_a_line_without_cells_reads_its_mean_token():
    built = model()
    expression, cells, mask = inputs()
    mask[1] = False  # the second line has no cell
    u = built.gene_vectors()[:-1]
    mean, tokens = built.tokens(expression, cells)
    mixed = built.embed(u, mean, tokens, mask)
    alone = built.embed(u, mean, None, None)
    assert torch.allclose(mixed[1], alone[1], atol=1e-6)
    assert not torch.allclose(mixed[0], alone[0], atol=1e-3)
    assert torch.isfinite(mixed).all()


def test_pair_score_is_the_same_for_both_orders_and_ignores_padding():
    built = model()
    expression, cells, mask = inputs(lines=1)
    u = built.gene_vectors()
    a, b = torch.tensor([0, 2, -1, 5]), torch.tensor([1, 4, 3, 5])
    flag = torch.tensor([True, True, False, True])
    mean, tokens = built.tokens(expression, cells)
    forward = built.pair(u[a], u[b], flag, mean[0], tokens[0], mask[0])
    backward = built.pair(u[b], u[a], flag, mean[0], tokens[0], mask[0])
    assert torch.allclose(forward, backward, atol=1e-5)
    padded = torch.cat([tokens[0], torch.randn(3, WIDTH)])
    padded_mask = torch.cat([mask[0], torch.zeros(3, dtype=torch.bool)])
    assert torch.allclose(
        forward, built.pair(u[a], u[b], flag, mean[0], padded, padded_mask), atol=1e-5
    )
    without = built.pair(u[a], u[b], flag, mean[0], None, None)
    assert torch.isfinite(without).all() and not torch.allclose(forward, without)


def test_pair_score_stays_finite_when_two_genes_read_different_cells():
    """Sharp weights on different cells: the product of the two genes' weights
    underflows to zero in every cell, and the pair's weights must still be finite."""
    built = model()
    with torch.no_grad():
        built.query.weight.copy_(torch.eye(WIDTH) * 400.0)
    cells = torch.zeros(50, WIDTH)
    cells[0, 0] = cells[1, 1] = 4.0
    u_a, u_b = torch.zeros(1, WIDTH), torch.zeros(1, WIDTH)
    u_a[0, 0] = u_b[0, 1] = 4.0
    mask = torch.ones(50, dtype=torch.bool)
    alpha_a = built.weights(u_a, cells[None], mask[None])[0]
    alpha_b = built.weights(u_b, cells[None], mask[None])[0]
    assert float((alpha_a * alpha_b).sum().detach()) == 0.0  # it underflows
    score = built.pair(
        u_a, u_b, torch.ones(1, dtype=torch.bool), torch.zeros(WIDTH), cells, mask
    )
    assert torch.isfinite(score).all()


def test_line_term_starts_at_zero_and_is_the_same_for_both_orders():
    built = model()
    expression, cells, mask = inputs(lines=1)
    u = built.gene_vectors()
    a, b = torch.tensor([0, 2, -1, 5]), torch.tensor([1, 4, 3, 5])
    flag = torch.tensor([True, True, False, True])
    mean, tokens = built.tokens(expression, cells)
    own = torch.randn(GENES + 1, 3)
    line = (mean[0], tokens[0], mask[0])
    start = built.line_term(
        u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, NO_PAIR, *line
    )
    assert start.shape == (4,) and torch.all(start == 0)
    torch.nn.init.normal_(built.line_head[-1].weight, std=0.5)
    forward = built.line_term(
        u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, NO_PAIR, *line
    )
    backward = built.line_term(
        u[b], u[a], flag, {"own": own[b]}, {"own": own[a]}, NO_PAIR, *line
    )
    assert forward.abs().sum() > 0 and torch.allclose(forward, backward, atol=1e-5)
    other = built.line_term(
        u[a], u[b], flag, {"own": own[b]}, {"own": own[b]}, NO_PAIR, *line
    )
    assert not torch.allclose(forward, other)  # the gene-in-line features are read
    no_cells = built.line_term(
        u[a],
        u[b],
        flag,
        {"own": own[a]},
        {"own": own[b]},
        NO_PAIR,
        mean[0],
        None,
        None,
    )
    assert torch.isfinite(no_cells).all()
    with_line = {id(p) for p in built.line_parameters()}
    assert with_line and not with_line & {id(p) for p in built.backbone()}
    assert not with_line & {id(p) for p in built.head.parameters()}


def test_the_genes_input_reads_no_line_token_and_no_gene_deviation():
    built = model(line_input="genes")
    torch.nn.init.normal_(built.line_head[-1].weight, std=0.5)
    expression, cells, mask = inputs(lines=1)
    a, b = torch.tensor([0, 2, -1, 5]), torch.tensor([1, 4, 3, 6])
    flag = torch.tensor([True, True, False, True])
    own = torch.randn(GENES + 1, 3)
    mean, tokens = built.tokens(expression, cells)

    def term(mean: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
        u = built.line_vectors()
        return built.line_term(
            u[a],
            u[b],
            flag,
            {"own": own[a]},
            {"own": own[b]},
            NO_PAIR,
            mean,
            tokens[0],
            mask[0],
        )

    base = term(mean[0], own)
    assert base.abs().sum() > 0
    assert torch.allclose(base, term(mean[0] + 3.0, own))  # another mean token
    with torch.no_grad():
        built.deviation.add_(1.0)
    assert torch.allclose(base, term(mean[0], own))  # another gene deviation
    moved = own.clone()
    moved[0] += 2.0
    assert not torch.allclose(base, term(mean[0], moved))  # a gene-in-line feature


def test_the_genes_input_is_the_same_for_both_orders_and_reads_a_zero_anchor():
    built = model(line_input="genes")
    torch.nn.init.normal_(built.line_head[-1].weight, std=0.5)
    u = built.line_vectors()
    assert u.shape == (GENES + 1, WIDTH) and torch.all(u[-1] == 0)
    a, b = torch.tensor([0, 2, -1, 5]), torch.tensor([1, 4, 3, 6])
    flag = torch.tensor([True, True, False, True])
    own = torch.randn(GENES + 1, 3)
    own[-1] = 0.0
    mean = torch.randn(WIDTH)
    forward = built.line_term(
        u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, NO_PAIR, mean, None, None
    )
    backward = built.line_term(
        u[b], u[a], flag, {"own": own[b]}, {"own": own[a]}, NO_PAIR, mean, None, None
    )
    assert torch.isfinite(forward).all()
    assert torch.allclose(forward, backward, atol=1e-5)


def test_an_unknown_line_input_is_refused():
    with pytest.raises(ValueError, match="line_input"):
        model(line_input="cells")


def pair_world():
    """Five genes and the row of "no gene". Gene 0 has the paralogs 1 and 2, gene 1
    the paralog 0; genes 0 and 1 share the complex partner 3."""
    paralogs = torch.tensor([[1, 2], [0, 5], [5, 5], [5, 5], [5, 5], [5, 5]])
    member = torch.zeros(6, 6, dtype=torch.bool)
    for g, h in ((0, 1), (0, 2), (0, 3), (1, 3)):
        member[g, h] = member[h, g] = True
    partners = torch.tensor(
        [[1, 2, 3], [0, 3, 5], [0, 5, 5], [0, 1, 5], [5, 5, 5], [5, 5, 5]]
    )
    return {
        "value": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.0]),
        "silenced": torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        "present": torch.tensor([True, True, True, True, True, False]),
        "z": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.0]),
        "paralogs": paralogs,
        "partners": partners,
        "member": member,
    }


def test_pair_in_line_reads_the_other_paralogs_and_the_shared_partners():
    world = pair_world()
    a, b = torch.tensor([0, 1, -1, 0]), torch.tensor([1, 0, 3, 3])
    flag = torch.tensor([True, True, False, False])
    out = pair_in_line(a, b, flag, **world)
    assert out.shape == (4, PAIR_WIDTH)
    # (0, 1): gene 0 keeps the paralog 2 (value 3), gene 1 keeps none; they share
    # the partner 3 (value 4, silenced, z 0.4); gene 0 is silenced, z of gene 1 is 0.2
    expected = [3, 3, 3, 3, 1, 4, 1, 0.4, 1, 0.2, 0, 0]
    assert torch.allclose(out[0], torch.tensor(expected))
    assert torch.equal(out[0], out[1])  # the same for (B, A)
    # an anchor that is no gene: only z of gene B, in the anchor column
    assert torch.allclose(out[2], torch.tensor([0.0] * 11 + [0.4]))
    # an anchor row of the silenced gene 0 and gene 3: silenced(A) z(B) and z(B)
    assert torch.allclose(out[3, 9:], torch.tensor([0.4 + 0.1, 0.4, 0.4]))
    # gene 0's paralogs other than 3 are 1 and 2: the largest 3, the mean 2.5
    assert out[3, 0] == 3 and out[3, 2] == 2.5 and out[3, 4] == 1
    # a paralog that the line does not measure does not count
    world["present"][2] = False
    absent = pair_in_line(a[:1], b[:1], flag[:1], **world)
    assert torch.all(absent[0, :5] == 0)


def test_the_line_head_reads_the_pair_in_line_features():
    built = model(line_input="genes", pair_width=PAIR_WIDTH)
    torch.nn.init.normal_(built.line_head[-1].weight, std=0.5)
    u = built.line_vectors()
    a, b = torch.tensor([0, 2]), torch.tensor([1, 4])
    flag = torch.tensor([True, True])
    own = torch.randn(GENES + 1, 3)
    pair = torch.randn(2, PAIR_WIDTH)

    def term(pair: torch.Tensor) -> torch.Tensor:
        return built.line_term(
            u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, pair,
            torch.zeros(WIDTH), None, None,
        )  # fmt: skip

    assert not torch.allclose(term(pair), term(pair + 1.0))
