from gw_simulator.generate import run_sim
from gw_simulator.generate import default_simulator as defaultsim
import torch


def test_simple_sim():

    batchsize = 1
    theta = torch.ones((2,))

    masses, xs = run_sim(theta)

    assert masses.shape == theta.shape
    assert xs.shape != theta.shape
    assert list(xs.shape) == [1, 2, 8192]


def test_batched_sim():

    batchsize = 3
    theta = torch.ones((batchsize, 2))
    theta[..., 0] = torch.arange(1, batchsize+1)
    theta[..., 1] = theta[..., 0]*10

    masses, xs = run_sim(theta)

    assert masses.shape == theta.shape
    assert xs.shape != theta.shape
    assert list(xs.shape) == [batchsize, 2, 8192]
    assert not torch.allclose(xs[0], xs[1])
