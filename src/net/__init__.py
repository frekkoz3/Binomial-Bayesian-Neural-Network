r"""
██████  ██████  ██████  ███    ██ 
     ██ ██   ██      ██ ████   ██ 
 █████  ██████   █████  ██ ██  ██ 
██      ██   ██ ██      ██  ██ ██ 
███████ ██████  ███████ ██   ████                                 
"""
from torch import nn
from enum import Enum


DEFAULT_CONFIG = {
        "input_dim": None,
        "output_dim": None,
        "n_hidden_layer": 2,
        "hidden_dims": [10000, 1],
        "bias": True,
        "min_val" : -1,
        "max_val" : 1,
        "N" : 50,
        "activations": "relu",
        "device": "cpu",
}

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "identity": nn.Identity,
}


class Mode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"

class BaseMLP(nn.Module):

    def forward(self, x):
        raise NotImplementedError
    
    def regularization_loss(self):
        return 0.0
