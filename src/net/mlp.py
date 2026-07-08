r"""
██████  ██████  ██████  ███    ██ 
     ██ ██   ██      ██ ████   ██ 
 █████  ██████   █████  ██ ██  ██ 
██      ██   ██ ██      ██  ██ ██ 
███████ ██████  ███████ ██   ████                                 
"""

import torch
from torch import nn

from src.net import *

class MLP(BaseMLP):

    def __init__(self, config: dict | None = None, *args, **kwargs):
        """
            Config is a dictionary containing:
            "input_dim" : int | None = None,
            "output_dim" : int | None = None,
            "n_hidden_layer" : int = 1,
            "hidden_dims" : int | list[int] = 128, (if is instance of int, the same dimension will be used across all the layers)
            "bias" : bool = True,
            "activations" : str | list["str"] = "relu", (if is instance of str, the same activation will be used across all the layers)
        """
        super().__init__(*args, **kwargs)

        cfg = DEFAULT_CONFIG.copy()
        if config is not None:
            cfg.update(config) # Merging

        # Required parameters
        if cfg["input_dim"] is None:
            raise ValueError("'input_dim' must be specified.")
        if cfg["output_dim"] is None:
            raise ValueError("'output_dim' must be specified.")

        n_hidden = cfg["n_hidden_layer"]
        self.heteroscedastic = cfg["heteroscedastic"]

        # Normalize hidden_dims
        if isinstance(cfg["hidden_dims"], int):
            hidden_dims = [cfg["hidden_dims"]] * n_hidden
        else:
            hidden_dims = list(cfg["hidden_dims"])
            if len(hidden_dims) != n_hidden:
                raise ValueError(
                    f"'hidden_dims' must have length {n_hidden}, "
                    f"got {len(hidden_dims)}."
                )

        # Normalize activations
        if isinstance(cfg["activations"], str):
            activations = [cfg["activations"]] * n_hidden
        else:
            activations = list(cfg["activations"])
            if len(activations) != n_hidden:
                raise ValueError(
                    f"'activations' must have length {n_hidden}, "
                    f"got {len(activations)}."
                )

        layers = []

        in_dim = cfg["input_dim"]
        for h_dim, act_name in zip(hidden_dims, activations):
            layers.append(nn.Linear(in_dim, h_dim))

            if act_name not in ACTIVATIONS:
                raise ValueError(f"Unknown activation '{act_name}'.")

            layers.append(ACTIVATIONS[act_name]())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, cfg["output_dim"]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    batch_size = 1
    input_dim = 2
    output_dim = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(size=[batch_size, input_dim]).to(device)

    cfg = {"input_dim" : input_dim, "output_dim" : output_dim}

    model = MLP(cfg).to(device)

    print(model(x))

        