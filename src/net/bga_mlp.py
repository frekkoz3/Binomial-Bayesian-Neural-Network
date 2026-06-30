r"""
██████  ██████  ██████  ███    ██ 
     ██ ██   ██      ██ ████   ██ 
 █████  ██████   █████  ██ ██  ██ 
██      ██   ██ ██      ██  ██ ██ 
███████ ██████  ███████ ██   ████                                 
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.net import *

class BinomialGaussianLinear(nn.Module):
    """
        Binomial linear layer using Gaussian approximation of a Binomial distribution.

        Each weight is modeled as a learnable Binomial random variable

            X ~ Binomial(N, p),

        where the probability parameter is obtained from an unconstrained learnable  parameter `rho` through the logistic function

            p = sigmoid(rho).

        For sufficiently large values of `N`, the Binomial distribution is approximated using its Gaussian approximation

            X ≈ Np + sqrt(Np(1-p)) * eps,
            eps ~ N(0, 1),

        and the sampled value is discretized using a rounding operation.
        Since `round(eps)` is not differentiable, the Straight-Through Estimator (STE) is used during backpropagation:

            x_continuous = Np + sqrt(Np(1-p)) * eps
            x_round = round(x_continuous)
            x = x_continuous + (x_round - x_continuous).detach()

        This preserves the rounded value during the forward pass while propagating the identity gradient during the backward pass.

        The discrete Binomial support {0, ..., N} is then mapped to an arbitrary interval [min, max] through the affine transformation

            w = min + X * (max - min) / N.

        As an example, choosing

            min = -1
            max = 1
            N = 4

        produces the set of admissible weights

            {-1, -0.5, 0, 0.5, 1},

        since a Binomial distribution with parameter `N` has a support of size `N + 1`.

        The learnable parameter `rho` is initialized from a uniform distribution in [0, 1].
        An optional bias term can also be included; when enabled, the bias is modeled by an independent Binomial distribution
        with its own learnable parameters.
    """

    def __init__(self, in_features, out_features, min_val : int = -5, max_val : int = 5, N : int = 50, bias=True):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val
        self.N = N

        self.avg_inference = False

        # Unconstrained parameter -> p = sigma(rho)
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        if bias:
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_rho", None)

        self.reset_parameters()

    def _set_avg_inference(self, flag : bool = True):
        self.avg_inference = flag

    def reset_parameters(self):
        nn.init.uniform_(self.weight_rho, -1, 1)

        if self.bias_rho is not None:
            nn.init.uniform_(self.bias_rho, -1, 1)

    def forward(self, x):
        p = torch.sigmoid(self.weight_rho)

        if self.avg_inference:
            # use model._set_avg_inference(True) to use this modality instead
            # this modality simply let you use the average output of the distribution as weight
            # instead of sampling from the actual distribution
            w = p*self.N
            if self.bias_rho:
                p_b = torch.sigmoid(self.bias_rho)
                b = p_b*self
            else:
                b = None
            return F.Linear(x, w, b)

        # Gaussian approximation
        mu = self.N * p
        sigma = torch.sqrt(self.N * p * (1 - p) + 1e-8)

        eps = torch.randn_like(mu)
        w = mu + sigma * eps

        # optional discretization (STE)
        w_round = w.round()
        w = w + (w_round - w).detach()

        # map to [min, max]
        w = self.min_val + (self.max_val - self.min_val) * (w / self.N)

        if self.bias_rho is not None:
            p_b = torch.sigmoid(self.bias_rho)
            mu_b = self.N * p_b
            sigma_b = torch.sqrt(self.N * p_b * (1 - p_b) + 1e-8)

            eps_b = torch.randn_like(mu_b)
            b = mu_b + sigma_b * eps_b

            b_round = b.round()
            b = b + (b_round - b).detach()

            b = self.min_val + (self.max_val - self.min_val) * (b / self.N)
        else:
            b = None

        return F.linear(x, w, b)
    
class BGA_MLP(BaseMLP):
    """
        Multi Layer Perceptron (MLP) with Binomial distribution over the weights (approximated as Gaussian).
        Uses reparametrization-trick for backprop.
    """

    def __init__(self, config : dict | None = None, *args, **kwargs):
        """
            Config is a dictionary containing:
            "input_dim" : int | None = None,
            "output_dim" : int | None = None,
            "n_hidden_layer" : int = 1,
            "hidden_dims": int | list[int] = 128, (if is instance of int, the same dimension will be used across all the layers)
            "bias" : bool = True,
            "min_val" : int = -5,
            "max_val" : int = 5,
            "N" : int = 50,
            "activations": str | list["str"] = "relu", (if is instance of str, the same activation will be used across all the layers)
            "device": str = "cuda" (possible values : "cpu", "cuda", "xpu"),
        """
        super().__init__(*args, **kwargs)

        cfg = DEFAULT_CONFIG.copy()
        if config is not None:
            cfg.update(config) #Mergin 

        # Required parameters
        if cfg["input_dim"] is None:
            raise ValueError("'input_dim' must be specified.")
        if cfg["output_dim"] is None:
            raise ValueError("'output_dim' must be specified.")

        n_hidden = cfg["n_hidden_layer"]

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
            layers.append(BinomialGaussianLinear(
                                            in_features=in_dim, 
                                            out_features=h_dim,
                                            min_val=cfg["min_val"],
                                            max_val=cfg["max_val"],
                                            N=cfg["N"],
                                            bias=cfg["bias"])
                                        )

            if act_name not in ACTIVATIONS:
                raise ValueError(f"Unknown activation '{act_name}'.")

            layers.append(ACTIVATIONS[act_name]())
            in_dim = h_dim

        layers.append(BinomialGaussianLinear(hidden_dims[-1],
                                             cfg["output_dim"],
                                             min_val=cfg["min_val"],
                                             max_val=cfg["max_val"],
                                             N=cfg["N"],
                                             bias=cfg["bias"])
                        )

        self.model = nn.Sequential(*layers)
        self.to(cfg["device"])

    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':

    batch_size = 10
    input_dim = 4
    output_dim = 10
    device = "cpu"

    x = torch.randn(size=[batch_size, input_dim]).to(device)

    cfg = {"input_dim" : input_dim, "output_dim" : output_dim, "device" : device}

    model = BGA_MLP(cfg)

    print(model(x))