from gw_simulator.generate import run_sim
import torch


def test_simple_sim():

    batchsize = 1
    theta = torch.ones((2,))

    masses, xs = run_sim(theta)
