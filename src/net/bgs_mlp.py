"""
██████  ██████  ██████  ███    ██
     ██ ██   ██      ██ ████   ██
 █████  ██████   █████  ██ ██  ██
██      ██   ██ ██      ██  ██ ██
███████ ██████  ███████ ██   ████
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from src.net import *



class TauScheduler:
    """
    A generic class that implements a tau scheduling strategy for the Gumbel-Softmax reparameterization trick.
    Currently implemented annelaing strategies are:
        - exponential: tau = tau_0 * exp(-k * epoch)
        - linear: tau = max(tau_min, tau_0 - k * epoch)
    """
    def __init__(self):
        pass
    def tau(self):
        pass
    def step(self):
        pass



class ExponentialTauScheduler(TauScheduler):
    """
    Exponential annealing strategy for tau scheduling.
    tau = tau_0 * exp(-k * epoch)
    """
    def __init__(self, tau_0 : float = 1.0, k : float = 0.01):
        super().__init__()
        self.tau_0 = tau_0
        self.k = k
        self.epoch = 0

    @property
    def tau(self):
        """Returns the current Tau value"""
        return self.tau_0 * np.exp(-self.k * self.epoch)

    def step(self):
        """Increments epoch counter for tau scheduling"""
        self.epoch += 1



class LinearTauScheduler(TauScheduler):
    """
    Linear annealing strategy for tau scheduling.
    tau = max(tau_min, tau_0 - k * epoch)
    """
    def __init__(self, tau_0 : float = 1.0, k : float = 0.01, tau_min : float = 0.1):
        super().__init__()
        self.tau_0 = tau_0
        self.k = k
        self.tau_min = tau_min
        self.epoch = 0

    @property
    def tau(self):
        """Returns the current Tau value"""
        return max(self.tau_min, self.tau_0 - self.k * self.epoch)

    def step(self):
        """Increments epoch counter for tau scheduling"""
        self.epoch += 1



class BinomialGumbelSoftmaxLinear(nn.Module):
    """
    Binomial linear layer using Gumbel-Softmax approximation of a binomial distribution.

    Each weight is modeled as a learnable binomial random variable
        w ~ Binomial (N, p)
    where `p` is obtained from an unconstrained learnable parameter `rho` through the logistic function to stay in the
    valid range [0, 1], and `N` is fixed.

    Since doing automatic differentiation is impossible in this discrete scenario, we apply here the Gumbel-Softmax (GS)
    reparameterization trick [1][2].
    In its original formalization, GS is intended to be used only for bernoullian and categorical variables.
    However, recently [3] stated that any discrete distribution with finite support can be approximated by the GS
    reparameterization trick and to deal our binomial distribution as it was categorical.
    The bounded values support {0, ..., N} allows also avoiding the truncation step as proposed in [3].

    The GS trick consists in what follows.
    Suppose to have a categorical distribution W~Cat(1, N), with class probabilities pi = {pi_1, pi_2, ..., pi_N}.
    Suppose also that each pi_i is one-hot encoded, so pi_i lies in {0,1}^N.
    1. We sample a parameter u_i ~ Uniform(0, 1)
    2. We generate an auxiliar Gumbel(0,1) sample `g_i` as:
        g_i = - log( - log(u_i) )
    3. We apply the softmax function to g:
        s_i ≈ softmax( (g_i+log(pi_i))/tau  )
       where `tau` is a temperature parameter representing a trade-off between the quality of the approximation (tau=0)
       and the amount of variance we introduce (tau=infinite leads to a uniform distribution in {0,N}).
       Usually, `tau` is annealed, thus tau(epoch).
    4. Finally, we remap the resulting value to the original support {0, ..., N} as:
        w_i = sum_{j=0}^{N} s_i * j

    Notice how, after this scheme, w_i becomes fully differentiable.
    To preserve the binomiality, we apply the straight-through estimator trick, i.e.:
        w_i = w_i + (round(w_i) - w_i).detach()
    That applies the discretization step in the forward pass, but allows the gradient to flow through the original `w_i`
    in the backward pass.

    Finally, the resulting weight is remapped to the arbitrary interval {min_val, max_val} as explained in the
    bga_mlp.BinomialGaussianLinear description.


    References
    ----------
    [1] Jang, E., et al., 2016, "Categorical reparameterization with gumbel-softmax."
    [2] Maddison, C., et al., 2016,  "The concrete distribution: A continuous relaxation of discrete random variables."
    [3] Joo, W., et al., 2025, "Generalized Gumbel-Softmax gradient estimator for generic discrete random variables."
    """

    def __init__(self,
                 in_features : int,
                 out_features : int,
                 min_val : int = -5,
                 max_val : int = 5,
                 N : int = 50,
                 bias : bool = True,
                 tau_scheduler : str | None = None,
                 tau_parameters : dict | None = None):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val
        self.N = N

        self.tau_scheduler = eval(tau_scheduler)(**tau_parameters if tau_parameters else {}) if tau_scheduler else 1.

        self.rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else self.register_parameter("bias", None)

        self._reset_parameters()


    def _reset_parameters(self):
        """Initialize the learnable parameters"""
        nn.init.uniform_(self.rho, -1, 1)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -1, 1)


    def _get_tau(self):
        """Get tau value"""
        return self.tau_scheduler.tau if isinstance(self.tau_scheduler, TauScheduler) else 1


    def forward_param(self, param):
        """Forward definition for a generic parameter"""
        p = torch.sigmoid(param)
        tau = self._get_tau()

        # Compute the class probabilities for the binomial distribution
        pi = torch.zeros((self.N+1, *p.shape), device=p.device)
        for k in range(self.N+1):
            pi[k] = math.comb(self.N, k) * (p**k) * ((1-p)**(self.N-k))
        pi = torch.clamp(pi, min=1e-10)

        # Apply Gumbel-Softmax reparameterization trick
        u = torch.rand_like(pi)
        g = - torch.log( - torch.log(u+1e-10) + 1e-10)
        soft = F.softmax( (g+torch.log(pi))/tau, dim=0 )

        # Soft is one-hot encoded, so we need to remap it to the original values {0, ..., N}
        categories = torch.arange(self.N + 1, device=p.device).view(-1, *([1]*len(p.shape)))
        param = torch.sum(soft * categories, dim=0)

        # Discretization
        param_round = param.round()
        param = param + (param_round - param).detach()

        # Map to [min_val, max_val]
        param = self.min_val + (self.max_val - self.min_val) * param / self.N

        return param


    def forward(self, x):
        """Forward pass"""
        w = self.forward_param(self.rho)
        b = self.forward_param(self.bias) if self.bias is not None else None

        self.tau_scheduler.step()
        return F.linear(x, w, b)



class BGS_MLP(nn.Module):

    """
    Multi Layer Perceptron (MLP) with Binomial Gumbel-Softmax distribution over the weights.
    Uses the BinomialGumbelSoftmaxLinear layer to model the weights of each layer as a binomial random variable,
    and applies the Gumbel-Softmax reparameterization trick to allow for backpropagation through the discrete weights.
    """


    def __init__(self,
                 config : dict | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        cfg = DEFAULT_CONFIG.copy()
        if config is not None:
            cfg.update(config)

        if cfg["input_dim"] is None or cfg["output_dim"] is None:
            raise ValueError("'input_dim' and 'output_dim' must be specified.")

        n_hidden_layers = cfg["n_hidden_layer"]
        input_dims = cfg["input_dim"]
        hidden_dims = [cfg["hidden_dims"]] * n_hidden_layers if isinstance(cfg["hidden_dims"], int) else list(cfg["hidden_dims"])
        activations = [cfg["activations"]] * n_hidden_layers if isinstance(cfg["activations"], str) else list(cfg["activations"])

        layers = []
        for hidden_dim, activation in zip(hidden_dims, activations):
            layers.append( BinomialGumbelSoftmaxLinear(in_features=input_dims,
                                                      out_features=hidden_dim,
                                                      min_val=cfg["min_val"],
                                                      max_val=cfg["max_val"],
                                                      N=cfg["N"],
                                                      bias=cfg["bias"],
                                                      tau_scheduler=cfg.get("tau_scheduler", None)) )
            layers.append(ACTIVATIONS[activation]())
            input_dims = hidden_dim

        layers.append( BinomialGumbelSoftmaxLinear(in_features=input_dims,
                                                   out_features=cfg["output_dim"],
                                                   min_val=cfg["min_val"],
                                                   max_val=cfg["max_val"],
                                                   N=cfg["N"],
                                                   bias=cfg["bias"],
                                                   tau_scheduler=cfg.get("tau_scheduler", None)) )
        self.model = nn.Sequential(*layers)
        self.to(cfg["device"])


    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":

    batch_size = 10
    input_dim = 4
    output_dim = 10
    device = "cpu"

    x = torch.randn(size=[batch_size, input_dim]).to(device)

    cfg = {"input_dim" : input_dim, "output_dim" : output_dim, "device" : device, "tau_scheduler": "ExponentialTauScheduler", "tau_parameters" : {"tau_0" : 1.0, "k" : 0.01}}

    model = BGS_MLP(cfg)

    print(model(x))
