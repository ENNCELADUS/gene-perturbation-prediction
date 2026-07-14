import torch

from aivc_model.response import ResponseEncoder, TrainableDiagonalGMM


def test_response_encoder_is_shared_linear_2000_to_128() -> None:
    encoder = ResponseEncoder(2000, 128)
    output = encoder(torch.randn(7, 2000))
    assert output.shape == (7, 128)
    assert encoder.linear.in_features == 2000
    assert encoder.linear.out_features == 128


def test_trainable_gmm_parameters_receive_finite_gradients() -> None:
    torch.manual_seed(42)
    gmm = TrainableDiagonalGMM(128, 8, 1e-4, 0.02)
    bag = torch.randn(16, 128, requires_grad=True)
    control = torch.randn(16, 128, requires_grad=True)
    loss = gmm(bag, control).square().mean() + gmm.negative_log_likelihood(bag)
    loss.backward()
    for parameter in (gmm.means, gmm.raw_variances, gmm.mixture_logits):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_control_occupancy_is_recomputed_after_gmm_update() -> None:
    torch.manual_seed(42)
    gmm = TrainableDiagonalGMM(4, 3, 1e-4, 0.02)
    bag = torch.randn(6, 4)
    control = torch.randn(6, 4)
    before = gmm(bag, control).detach().clone()
    with torch.no_grad():
        gmm.means.add_(0.5)
    after = gmm(bag, control).detach()
    assert not torch.equal(before, after)
