import torch

from sl_dl_model.pair_head import SymmetricPairHead


def test_pair_head_output_shape():
    head = SymmetricPairHead(emb_dim=12, hidden=(16,))
    e_a = torch.randn(4, 12)
    e_b = torch.randn(4, 12)
    ge = torch.randn(4, 5)
    logit = head(e_a, e_b, ge)
    assert logit.shape == (4,)


def test_pair_head_is_swap_invariant():
    torch.manual_seed(0)
    head = SymmetricPairHead(emb_dim=12, hidden=(16,)).eval()
    e_a = torch.randn(4, 12)
    e_b = torch.randn(4, 12)
    # GeneEffect block must also be swap-invariant; reuse same ge for both orders
    ge = torch.randn(4, 5)
    with torch.no_grad():
        ab = head(e_a, e_b, ge)
        ba = head(e_b, e_a, ge)
    assert torch.allclose(ab, ba, atol=1e-5)
