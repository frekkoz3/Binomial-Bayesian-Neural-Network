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
        "hidden_dims": [10, 10],
        "bias": True,
        "min_val" : -2,
        "max_val" : 2,
        "N" : 50,
        "activations" : "relu",
        "device" : "cuda",
        "resolution" : 8
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
    
class Mode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"
