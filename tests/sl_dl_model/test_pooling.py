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
