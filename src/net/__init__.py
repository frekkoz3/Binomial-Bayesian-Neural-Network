from torch import nn

DEFAULT_CONFIG = {
        "input_dim": None,
        "output_dim": None,
        "n_hidden_layer": 1,
        "hidden_dims": 128,
        "bias": True,
        "min_val" : -5,
        "max_val" : 5,
        "N" : 50,
        "activations": "relu",
        "device": "cuda",
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