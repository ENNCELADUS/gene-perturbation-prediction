import torch

from src.model.response_mlp import ResponseMLP


def test_untrained_model_is_exactly_no_change():
    torch.manual_seed(0)
    model = ResponseMLP(cell_dim=7, gene_dim=5, hidden=16, out_dim=4)
    basal = torch.rand(3, 4)
    out = model(torch.randn(3, 7), basal, torch.randn(5))
    assert torch.equal(out, basal)


def test_gene_changes_output_after_one_step():
    torch.manual_seed(0)
    model = ResponseMLP(cell_dim=7, gene_dim=5, hidden=16, out_dim=4)
    cells, basal = torch.randn(3, 7), torch.rand(3, 4)
    loss = (model(cells, basal, torch.randn(5)) - 1.0).square().mean()
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    a = model(cells, basal, torch.ones(5))
    b = model(cells, basal, -torch.ones(5))
    assert not torch.allclose(a, b)


def test_per_cell_output_shape():
    model = ResponseMLP(cell_dim=7, gene_dim=5, hidden=16, out_dim=4)
    assert model(torch.randn(9, 7), torch.rand(9, 4), torch.randn(5)).shape == (9, 4)
