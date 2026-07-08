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
import math

from src.net import *

LOG_2PI = torch.tensor(math.log(2*math.pi))

class GaussianLinear(nn.Module):
    """
        Gaussian linear layer with learnable weight uncertainty.

        Each weight is modeled as a Gaussian random variable with learnable mean and
        variance:

            w ~ N(mu, sigma^2),

        where:

        - `mu` is a learnable parameter representing the mean
        - `rho` is an unconstrained learnable parameter used to parameterize the
        standard deviation
        - sigma is obtained via a positivity-preserving transformation:

            sigma = softplus(rho)

        This ensures that sigma > 0 for all values of rho.

        To enable backpropagation through stochastic sampling, the forward pass uses
        the reparameterization trick:

            eps ~ N(0, 1)
            w = mu + sigma * eps

        This separates the stochasticity (eps) from the learnable parameters (mu, rho),
        allowing gradients to flow through both.

        - `mu` is initialized using Kaiming uniform initialization
        - `rho` is initialized to a constant value corresponding to:

            sigma = softplus(-5)

        which corresponds to a small initial variance.

        An optional bias term can also be included; when enabled, the bias is
        modeled by an independent Gaussian distribution with its own learnable
        parameters.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # Mean parameters
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Unconstrained parameter -> sigma = softplus(rho)
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self.reset_parameters()
        self.kl = torch.tensor(0.)

        self.register_buffer("prior_mu", torch.tensor(0.))
        self.register_buffer("prior_sigma", torch.tensor(1.))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu)

        # Initial std
        nn.init.constant_(self.weight_rho, -5.)

        if self.bias_mu is not None:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_rho, -5.)

    def _gaussian_kl(self, mu_q, sigma_q, mu_p, sigma_p):
        """
            Closed form for the kl divergence between two gaussians
        """
        return (
            torch.log(sigma_p / sigma_q)
            + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)
            - 0.5
        )
    
    def forward(self, x):

        weight_sigma = F.softplus(self.weight_rho)

        eps_w = torch.randn_like(weight_sigma)
        weight = self.weight_mu + weight_sigma * eps_w
        self.kl = self._gaussian_kl(weight, mu1=self.weight_mu, sigma1=weight_sigma, mu2=self.prior_mu, sigma2=self.prior_sigma)

        if self.bias_mu is not None:
            bias_sigma = F.softplus(self.bias_rho)
            eps_b = torch.randn_like(bias_sigma)
            bias = self.bias_mu + bias_sigma * eps_b
            self.kl = self.kl + self._gaussian_kl(bias, mu1=self.bias_mu, sigma1=bias_sigma, mu2=self.prior_mu, sigma2=self.prior_sigma)
        else:
            bias = None

        return F.linear(x, weight, bias)



class G_MLP(BaseMLP):
    """
        Multi Layer Perceptron (MLP) with Gaussian distribution over the weights.
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
            "activations": str | list["str"] = "relu", (if is instance of str, the same activation will be used across all the layers)
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
            layers.append(GaussianLinear(in_dim, h_dim, cfg["bias"]))

            if act_name not in ACTIVATIONS:
                raise ValueError(f"Unknown activation '{act_name}'.")

            layers.append(ACTIVATIONS[act_name]())
            in_dim = h_dim

        layers.append(GaussianLinear(in_dim, cfg["output_dim"], bias=cfg["bias"]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def kl_loss(self):
        kl = 0.

        for module in self.modules():
            if isinstance(module, GaussianLinear):
                kl += module.kl

        return kl




if __name__ == '__main__':

    batch_size = 10
    input_dim = 4
    output_dim = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(size=[batch_size, input_dim]).to(device)

    cfg = {"input_dim" : input_dim, "output_dim" : output_dim}

    model = G_MLP(cfg).to(device)

    print(model(x))