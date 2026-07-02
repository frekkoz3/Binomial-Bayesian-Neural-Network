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


def normal_cdf(x, mu : int = 0, sigma : int = 1):
    z = (x - mu) / sigma
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))

def normal_icdf(p):
    return math.sqrt(2) * torch.erfinv(2*p - 1)

class BinomialGaussianLinear(nn.Module):
    """
        Binomial linear layer using Gaussian approximation of a Binomial distribution.

        Each weight is modeled as an affine transformation of a learnable Binomial random variable

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

        The learnable parameter `rho` is initialized from a gaussian with mean 0 and variance 1 (which contains mostly values in the range [-2.5, 2.5]).
        An optional bias term can also be included; when enabled, the bias is modeled by an independent Binomial distribution
        with its own learnable parameters.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 min_val : int = -5,
                 max_val : int = 5,
                 N : int = 50,
                 bias=True,
                 resolution : int = 8):
        """
            Notes : resolution parameters is related to the number of bit to use for storing the p parameters.
        """
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val
        self.N = N
        self.resolution = resolution

        self.avg_inference = False # the behavior of the inference change based on this.

        # Unconstrained parameter -> p = sigma(rho)
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        if bias:
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_rho", None)

        self.mode = "train"  # or "inference"
        self.register_buffer("weight_p", None)
        self.register_buffer("bias_p", None)

        self.reset_parameters()

    def get_extra_state(self):
        return {
            "min_val" : self.min_val,
            "max_val" : self.max_val,
            "N" : self.N,
            "resolution" : self.resolution
        }
    
    def set_extra_state(self, state):
        self.min_val = state["min_val"]
        self.max_val = state["max_val"]
        self.N = state["N"]
        self.resolution = state["resolution"]

    def _set_avg_inference(self, flag : bool = True):
        self.avg_inference = flag

    def _set_saving_mode(self, mode : Mode = Mode.TRAIN):
        """
            This function set the saving (and loading) mode.
            If the mode is set to "train" the rho tensor will be saved.
            If instead the mode is set to "inference" the pi tensore will be saved.
        """
        self.saving_mode = mode

    def reset_parameters(self, d : int | None = None):
        nn.init.normal_(self.weight_rho, std=0)

        if self.bias_rho is not None:
            nn.init.normal_(self.bias_rho, std = 0)

    def _expected_weight(self, p):
        return self.min_val + p * (self.max_val - self.min_val)

    def forward(self, x):
        if self.mode == Mode.INFERENCE:
            p = self.weight_p
            p_b = self.bias_p if self.bias_rho is not None else None
        else:
            p = torch.sigmoid(self.weight_rho)
            p_b = torch.sigmoid(self.bias_rho) if self.bias_rho is not None else None

        if self.avg_inference:
            # use model._set_avg_inference(True) to use this modality instead
            # this modality simply let you use the average output of the distribution as weight
            # instead of sampling from the actual distribution
            w = self._expected_weight(p)
            b = self._expected_weight(p_b) if p_b is not None else None

            return F.linear(x, w, b)

        mu = self.N * p
        sigma = torch.sqrt(self.N * p * (1 - p) + 1e-8)

        eps = torch.randn_like(mu)
        w = mu + sigma * eps

        w_round = w.round()
        w = w + (w_round - w).detach()

        w = self.min_val + (self.max_val - self.min_val) * (w / self.N)

        if p_b is not None:
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
    
    def export_inference_state(self, quantize=True):
        with torch.no_grad():
            p = torch.sigmoid(self.weight_rho)

            if quantize:
                step = (1<<self.resolution) - 1
                p = torch.round(p * step) / step

            state = {
                "min_val": self.min_val,
                "max_val": self.max_val,
                "N": self.N,
                "resolution": self.resolution,
                "weight_p": p.clone()
            }

            if self.bias_rho is not None:
                bp = torch.sigmoid(self.bias_rho)
                if quantize:
                    step = (1<<self.resolution) - 1
                    bp = torch.round(bp * step) / step
                state["bias_p"] = bp.clone()

            return state
    
    def load_inference_state(self, state):
        self.min_val = state["min_val"]
        self.max_val = state["max_val"]
        self.N = state["N"]
        self.resolution = state["resolution"]

        self.weight_p = state["weight_p"].clone()
        self.register_buffer("weight_p", self.weight_p)

        if "bias_p" in state:
            self.bias_p = state["bias_p"].clone()
            self.register_buffer("bias_p", self.bias_p)

        self.mode = Mode.INFERENCE

    def restore_train_mode(self):
        assert self.mode == Mode.INFERENCE

        eps = 1e-6
        p = torch.clamp(self.weight_p, eps, 1 - eps)

        self.weight_rho = nn.Parameter(torch.log(p/(1-p)))

        if self.bias_p is not None:
            bp = torch.clamp(self.bias_p, eps, 1 - eps)
            self.bias_rho = nn.Parameter(torch.log(bp/(1-bp)))

        self.weight_p = None
        self.bias_p = None

        self.mode =  Mode.TRAIN
    
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

        self.avg_inference = False

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def set_mode(self, mode: Mode = Mode.TRAIN):
        self.mode = mode
        for layer in self.model:
            if hasattr(layer, "mode"):
                layer.mode = mode

    def export_inference_state(self, quantize=True):
        state = []

        for layer in self.model:
            if hasattr(layer, "export_inference_state"):
                state.append(layer.export_inference_state(quantize))
            else:
                state.append(None)

        return {
            "layers": state
        }
    
    def load_inference_state(self, state):
        for layer, layer_state in zip(self.model, state["layers"]):
            if layer_state is not None and hasattr(layer, "load_inference_state"):
                layer.load_inference_state(layer_state)

        self.set_mode(Mode.INFERENCE)

    def restore_train_mode(self):
        for layer in self.model:
            if hasattr(layer, "restore_train_mode"):
                layer.restore_train_mode()

        self.set_mode(Mode.TRAIN)

    def set_avg_inference(self, flag : bool = True):
        for layer in self.model:
            if hasattr(layer, "_set_avg_inference"):
                layer._set_avg_inference(flag)
        
        self.avg_inference = flag
    
if __name__ == '__main__':

    # Parameters
    batch_size = 1
    input_dim = 10
    output_dim = 1
    device = "cpu"

    # Data
    x = torch.randn(size=[batch_size, input_dim])

    # Config
    cfg = {"input_dim" : input_dim, "output_dim" : output_dim}

    # Normal model
    model = BGA_MLP(cfg)

    # Avg inference set to true to visualize the model behavior
    # model.set_avg_inference(True)

    print(f"normal :\n{model(x)}")

    # Saving the model
    torch.save(model.state_dict(), "models/proof.pkl")

    # Loading the model
    state = torch.load("models/proof.pkl")

    model.load_state_dict(state)

    # Avg inference set to true to visualize the model behavior
    # model.set_avg_inference(True)

    print(f"loaded :\n{model(x)}")

    # Setting INFERENCE mode
    # model.set_mode(Mode.INFERENCE)

    # Saving the model in inference mode (quantized to 8 bit)
    torch.save(model.export_inference_state(), "models/8    bit_proof.pkl")

    # Loading the quantized model
    quant_state = torch.load("models/8bit_proof.pkl")

    model.load_inference_state(quant_state)

    # Avg inference set to true to visualize the model behavior
    model.set_avg_inference(True)

    print(f"quantized :\n{model(x)}")

    # Restore train mode (dequantize the model)
    model.restore_train_mode()
    
    # Avg inference set to true to visualize the model behavior
    # model.set_avg_inference(True)

    print(f"restored normal :\n{model(x)}")