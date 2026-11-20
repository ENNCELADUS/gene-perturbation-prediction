"""The gene-knowledge, cell-set model: the untrained model predicts the gene mean, the
pair score does not change with the order of the two genes, the order and the number
of the cells do not change the output, and a line without cells reads its mean token."""

from __future__ import annotations

import torch

from src.model.sl_embedding import WIDTH, GeneLineModel

GENES, EXPRESSION, CELL = 12, 6, 8


def model(seed: int = 0) -> GeneLineModel:
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
        gene_in_line={},
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
    start = built.line_term(u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, *line)
    assert start.shape == (4,) and torch.all(start == 0)
    torch.nn.init.normal_(built.line_head[-1].weight, std=0.5)
    forward = built.line_term(u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, *line)
    backward = built.line_term(
        u[b], u[a], flag, {"own": own[b]}, {"own": own[a]}, *line
    )
    assert forward.abs().sum() > 0 and torch.allclose(forward, backward, atol=1e-5)
    other = built.line_term(u[a], u[b], flag, {"own": own[b]}, {"own": own[b]}, *line)
    assert not torch.allclose(forward, other)  # the gene-in-line features are read
    no_cells = built.line_term(
        u[a], u[b], flag, {"own": own[a]}, {"own": own[b]}, mean[0], None, None
    )
    assert torch.isfinite(no_cells).all()
    with_line = {id(p) for p in built.line_parameters()}
    assert with_line and not with_line & {id(p) for p in built.backbone()}
    assert not with_line & {id(p) for p in built.head.parameters()}
