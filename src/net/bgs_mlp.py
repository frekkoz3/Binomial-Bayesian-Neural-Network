"""
██████  ██████  ██████  ███    ██
     ██ ██   ██      ██ ████   ██
 █████  ██████   █████  ██ ██  ██
██      ██   ██ ██      ██  ██ ██
███████ ██████  ███████ ██   ████
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from src.net import *
from src.net.weight_init import *



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



class ConstantTauScheduler(TauScheduler):
    """
    Constant tau scheduling strategy.
    tau = tau_0
    """
    def __init__(self, tau_0 : float = 1.0):
        super().__init__()
        self.tau_0 = tau_0
        self.epoch = 0

    @property
    def tau(self):
        """Returns the current Tau value"""
        return self.tau_0

    def step(self):
        """No-op for constant tau scheduling"""
        pass



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
                 tau_scheduler : str | None = None,
                 tau_parameters : dict | None = None,
                 resolution : int = 8,
                 bias : bool = True):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val
        self.N = N
        self.resolution = resolution

        self.tau_parameters = tau_parameters
        self.tau_scheduler = eval(tau_scheduler)(**tau_parameters if tau_parameters else {})

        self.mode = "train" # or "inference"
        self.avg_inference = False

        self.rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_p", None)
        self.register_buffer("bias_p", None)

        self._reset_parameters(d=in_features, o=out_features)

        # Log Combinations for Binomial Distribution:
        # log(N!) - log(k!) - log((N-k)!)
        k_tensor = torch.arange(self.N + 1, dtype=torch.float32)
        log_comb = torch.lgamma(torch.tensor(self.N + 1.0)) - torch.lgamma(k_tensor + 1.0) - torch.lgamma(self.N - k_tensor + 1.0)

        self.register_buffer("k_view", k_tensor.view(-1, 1, 1))
        self.register_buffer("log_comb_view", log_comb.view(-1, 1, 1))


    def _reset_parameters(self, d : int = None, o : int = None):
        """Initialize the learnable parameters"""
        std_sx = math.sqrt(weight_initialization(self.N, d, self.max_val, self.min_val))
        std_dx = math.sqrt(weight_initialization(self.N, o, self.max_val, self.min_val))
        std = (std_sx + std_dx) / 2
        print(f"Std SX: {std_sx}, Std DX: {std_dx}, Avg Std: {std}")
        nn.init.normal_(self.rho, std = std)
        nn.init.normal_(self.bias, std = std)


    def _get_tau(self):
        """Get tau value"""
        return self.tau_scheduler.tau


    def get_extra_state(self) -> Any:
        return {
            "min_val": self.min_val,
            "max_val": self.max_val,
            "N": self.N,
            "resolution": self.resolution,
            "tau_scheduler_name": self.tau_scheduler.__class__.__name__,
            "tau_parameters": self.tau_parameters,
            "tau_epoch": self.tau_scheduler.epoch
        }


    def set_extra_state(self, state: Any) -> None:
        self.min_val = state["min_val"]
        self.max_val = state["max_val"]
        self.N = state["N"]
        self.resolution = state["resolution"]
        self.tau_parameters = state["tau_parameters"]

        scheduler_name = state.get("tau_scheduler_name")
        if scheduler_name:
            scheduler_class = globals().get(scheduler_name)
            self.tau_scheduler = scheduler_class(**self.tau_parameters if self.tau_parameters else {})
            self.tau_scheduler.epoch = state.get("tau_epoch", 0)
        else:
            self.tau_scheduler = 1.


    def set_avg_inference(self, flag : bool = True):
        """Set the inference mode to average or not"""
        self.avg_inference = flag


    def _set_saving_mode(self, mode : Mode = Mode.TRAIN):
        """Set the saving mode to save the average or not"""
        self.saving_mode = mode


    def save_inference_state(self, quantize = True):
        """Save the inference state of the layer"""
        with torch.no_grad():
            p = torch.sigmoid(self.rho)

            if quantize:
                step = (1<<self.resolution) - 1
                p = torch.round(p * step) / step

            state = {
                "min_val": self.min_val,
                "max_val": self.max_val,
                "N": self.N,
                "resolution": self.resolution,
                "weight_p": p.clone(),
                "tau_scheduler_name": self.tau_scheduler.__class__.__name__,
                "tau_parameters": self.tau_parameters
            }

            if self.bias is not None:
                bp = torch.sigmoid(self.bias)
                if quantize:
                    step = (1<<self.resolution) - 1
                    bp = torch.round(bp * step) / step
                state["bias_p"] = bp.clone()

            return state


    def load_inference_state(self, state):
        """Load the inference state of the layer"""
        self.min_val = state["min_val"]
        self.max_val = state["max_val"]
        self.N = state["N"]
        self.resolution = state["resolution"]

        self.weight_p = state["weight_p"].clone()
        self.register_buffer("weight_p", self.weight_p)
        self.tau_parameters = state["tau_parameters"]

        scheduler_name = state.get("tau_scheduler_name")
        if scheduler_name:
            scheduler_class = globals().get(scheduler_name)
            self.tau_scheduler = scheduler_class(**self.tau_parameters if self.tau_parameters else {})
            self.tau_scheduler.epoch = state.get("tau_epoch", 0)

        if "bias_p" in state:
            self.bias_p = state["bias_p"].clone()
            self.register_buffer("bias_p", self.bias_p)

        self.mode = Mode.INFERENCE


    def restore_train_mode(self):
        assert self.mode == Mode.INFERENCE

        eps = 1e-6
        p = torch.clamp(self.weight_p, eps, 1 - eps)

        self.rho = nn.Parameter(torch.log(p/(1-p)))

        self.bias = nn.Parameter(torch.log(self.bias_p/(1-self.bias_p))) if self.bias is not None else None

        self.weight_p = None
        self.bias_p = None

        self.mode =  Mode.TRAIN


    def _sample_weight(self, p):
        """Sample a weight from the binomial distribution given the probability p"""
        return torch.distributions.binomial.Binomial(total_count=self.N, probs=p).sample()


    def _expected_weight(self, p):
        """Compute the expected weight value given the probability p"""
        return self.min_val + (self.max_val - self.min_val) * p / self.N


    def forward_param(self, param):
        """Forward definition for a generic parameter"""
        if self.mode == Mode.INFERENCE:
            p = self.weight_p if param is self.rho else self.bias_p
            weight = self._sample_weight(p)
            return self._expected_weight(weight)

        if self.avg_inference:
            param = self._expected_weight(torch.sigmoid(param))
            return param

        tau = self._get_tau()

        # Use log(p) and log(1-p) for numerical stability
        log_p = F.logsigmoid(param)
        log_1_minus_p = F.logsigmoid(-param)
        # Broadcast to get logits
        logits = self.log_comb_view + (self.k_view * log_p) + ((self.N - self.k_view) * log_1_minus_p)
        logits_t = logits.permute(1, 2, 0)

        # Apply Gumbel-Softmax reparameterization trick
        soft = F.gumbel_softmax(logits_t, tau=tau, hard=False, dim=-1)

        # Soft is one-hot encoded, so we need to remap it to the original values {0, ..., N}
        categories = torch.arange(self.N + 1, device=param.device, dtype=param.dtype)
        param = torch.sum(soft * categories, dim=-1)

        # Discretization
        param_round = param.round()
        param = param + (param_round - param).detach()

        # Map to [min_val, max_val]
        param = self.min_val + (self.max_val - self.min_val) * param / self.N

        return param


    def forward(self, x):
        """Forward pass"""
        w = self.forward_param(self.rho)
        # b = self.forward_param(self.bias) if self.bias is not None else None

        b = self.bias if self.bias is not None else None

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
                                                      bias = cfg["bias"],
                                                      tau_scheduler=cfg.get("tau_scheduler", None),
                                                      tau_parameters=cfg.get("tau_parameters", None)) )
            layers.append(ACTIVATIONS[activation]())
            input_dims = hidden_dim

        layers.append( BinomialGumbelSoftmaxLinear(in_features=input_dims,
                                                   out_features=cfg["output_dim"],
                                                   min_val=cfg["min_val"],
                                                   max_val=cfg["max_val"],
                                                   N=cfg["N"],
                                                   bias = cfg["bias"],
                                                   tau_scheduler=cfg.get("tau_scheduler", None),
                                                   tau_parameters=cfg.get("tau_parameters", None)) )
        self.model = nn.Sequential(*layers)
        self.to(cfg["device"])


        self.avg_inference = False
        self.mode = Mode.TRAIN


    def set_mode(self, mode : Mode = Mode.TRAIN):
        self.mode = mode
        for layer in self.model:
            if hasattr(layer, "mode"):
                layer.mode = mode


    def export_inference_state(self, quantize = True):
        """Export the inference state of the model"""
        state = []

        for layer in self.model:
            if hasattr(layer, "save_inference_state"):
                state.append(layer.save_inference_state(quantize=quantize))
            else:
                state.append(None)

        return {
            "layers": state,
        }


    def load_inference_state(self, state):
        """Load the inference state of the model"""
        for layer, layer_state in zip(self.model, state["layers"]):
            if hasattr(layer, "load_inference_state") and layer_state is not None:
                layer.load_inference_state(layer_state)

        self.set_mode(Mode.INFERENCE)


    def restore_train_mode(self):
        """Restore the model to training mode"""
        for layer in self.model:
            if hasattr(layer, "restore_train_mode"):
                layer.restore_train_mode()

        self.set_mode(Mode.TRAIN)


    def set_avg_inference(self, flag : bool = True):
        """Set the inference mode to average or not"""
        for layer in self.model:
            if hasattr(layer, "set_avg_inference"):
                layer.set_avg_inference(flag)

        self.avg_inference = flag

    def set_resolution(self, resolution : int):
        """Set the storage bit-width of `p` on every layer."""
        for layer in self.model:
            if hasattr(layer, "resolution"):
                layer.resolution = resolution


    def step_tau(self):
        """
        Advances the tau scheduler by one epoch for all discrete layers.
        Must be called once at the end of each training epoch.
        """
        for layer in self.model:
            if hasattr(layer, "tau_scheduler"):
                layer.tau_scheduler.step()

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":

    batch_size = 10
    input_dim = 4
    output_dim = 10
    device = "cpu"

    x = torch.randn(size=[batch_size, input_dim]).to(device)

    cfg = {"input_dim" : input_dim,
           "output_dim" : output_dim,
           "device" : device,
           "tau_scheduler": "ExponentialTauScheduler",
           "tau_parameters" : {"tau_0" : 1.0, "k" : 0.01}
           }

    model = BGS_MLP(cfg)

    # Normal model
    model = BGS_MLP(cfg)

    # Avg inference set to true to visualize the model behavior
    model.set_avg_inference(True)

    print(f"normal :\n{model(x)}")

    # Saving the model
    torch.save(model.state_dict(), "models/try.pkl")

    # Loading the model
    state = torch.load("models/try.pkl", weights_only=False)
    model.load_state_dict(state)

    # Avg inference set to true to visualize the model behavior
    model.set_avg_inference(True)
    print(f"loaded :\n{model(x)}")

    # Setting INFERENCE mode
    model.set_mode(Mode.INFERENCE)

    # Saving the model in inference mode (quantized to 8 bit)
    torch.save(model.export_inference_state(), "models/8bit_try.pkl")

    # Loading the quantized model
    quant_state = torch.load("models/8bit_try.pkl", weights_only=False)
    model.load_inference_state(quant_state)

    # Avg inference set to true to visualize the model behavior
    model.set_avg_inference(True)
    print(f"quantized :\n{model(x)}")

    # Restore train mode (dequantize the model)
    model.restore_train_mode()

    # Avg inference set to true to visualize the model behavior
    model.set_avg_inference(True)

    print(f"restored normal :\n{model(x)}")
