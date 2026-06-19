import torch

from sl_dl_model.pooling import MeanStdPool, build_pool, output_dim


def test_mean_std_pool_dim():
    pool = MeanStdPool()
    out = pool(torch.randn(20, 6))
    assert out.shape == (12,)


def test_build_pool_and_output_dim():
    assert output_dim("mean_std", 6) == 12
    pool = build_pool("mean_std", 6)
    assert isinstance(pool, MeanStdPool)


def test_unknown_pool_raises():
    try:
        build_pool("nope", 6)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_mean_std_pool_grad_finite_on_constant_feature():
    # std = sqrt(var); var=0 on a constant feature gives sqrt'(0)=inf grad (H3).
    bag = torch.zeros(10, 6, requires_grad=True)  # every feature constant
    out = MeanStdPool()(bag)
    out.sum().backward()
    assert torch.isfinite(bag.grad).all(), "pooling grad must be finite at var=0"
