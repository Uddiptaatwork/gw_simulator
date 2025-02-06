# Gravitational Wave Simulator

This package is meant to abstract the simulation of a gravitational wave event recorded by the LIGO experiment.

## Installation

Include the following in your `requirements.txt`:

``` 
https://github.com/psteinb/gw_simulator.git@master
```

or download and install the package:

``` bash
git clone https://github.com/psteinb/gw_simulator.git
cd gw_simulator
python -m pip install .
```

A pypi release is in the making.

## Usage

If you want to use the default simulator configs, do the following:

``` python
from gw_simulator.generate import run_sim
import torch

batchsize = 3
theta = torch.ones((batchsize, 2))
theta[..., 0] = torch.arange(1, batchsize+1)
theta[..., 1] = theta[..., 0]*10

masses, xs = run_sim(theta)
```

If you like to play with the simulator, you might want to do:

``` python
from gw_simulator.simulator.interface import GravitationalWaveBenchmarkSimulator as gws
import torch

batchsize = 3
theta = torch.ones((batchsize, 2))
theta[..., 0] = torch.arange(1, batchsize+1)
theta[..., 1] = theta[..., 0]*10

simulator = gws(path_to_config)
xs = simulator(theta)
```
