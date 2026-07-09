# Binomial Bayesian Neural Network

a project by Team Rocket - Bredariol, Riccio, Savorgnan

---

## The Idea

A **Binomial Bayesian Neural Network (2B2N)** is a Bayesian Neural Network whose weights are parameterized through a remapped Binomial distribution with a fixed number of trials $N$.

2B2N born as an experimental architecture with some nice (theoretical) properties:

* Natural Quantization
* Natural Maximum Variance
* Natural Discrete Weights

Discussion about this can be found in [this document](docs/notes.md).

We explored the architecture by changing the objective function, studying the variance at the initialization, trying different toy landscapes, evaluating the quantization capability and assessing the perfomance against other known architectures.

## Repository Structure

```bash
├── docs/                     # Folder containing paper references and our documentation for the project
├── models/                   # Folder containing trained models' weights
├── plots/                    # Folder containing plot from experiments
├── src/                      # Folder containing the code base
    ├── data/                 # Dataset module
    ├── exp/                  # Experiment scripts
    ├── net/                  # Networks modules
    ├── train/                # Training scripts
    ├── val/                  # Validation scripts
    └── vis/                  # Visualization module
└── README.md       
```

## Quick Setup

### Windows

```bash
py -3.12 -m venv .venv
.venv\scripts\activate
pip install -r requirements.txt
```

### Linux

```bash
py -3.12 -m venv .venv
source .venv\bin\activate
pip install -r requirements.txt
```

The actual state of the repository does not provides a common interface to use the modules from command lines.
If interested in using the BGS or the BGA modules you need to take them from their source code. [BGA implementation](src/net/bga_mlp.py) and [BGS implementation](src/net/bgs_mlp.py).
```bash
├─── docs/                    # Notes and References      
│    └─── notes.md            # Mathematical formulation and derivations of the 2B2N model
├─── models/                  # Trained models
├─── plots/                   # Plots and images used in the presentation
├─── src/                     # Source code
│    ├─── data/               # Dataset loading and preprocessing 
│    ├─── exp/                # Experimentation scripts. See files for details
│    ├─── net/                # Neural network architecture and layers
│    ├─── train/              # Training scripts
│    ├─── val/                # Validation scripts
│    └─── vis/                # Visualization tools
├─── .gitignore
├─── requirements.txt
└─── README.md
```

## Experiments

The section presents a brief description of the experiments conducted on 2B2N architecture.
All the scripts are located in the `src/exp` folder.

* `gap_behavior`: The goal of the experiment is to compare the behavior of different 2B2N architectures on a dataset with a gap in the middle of the input space.
  The models are trained without data in the gap, but evaluated on the whole input space. The experiment is useful to understand the generalization properties of the models and the behavior of the variance out of distribution.
* `impact_initialization`: The goal of the experiment is to analyze the impact of starting with a variance on the weights that is wide
  enough to allow effective learning but small enough to avoid the spread of the prediction variance. The experiment compares the behavior of 2B2N models with and without our weight initialization. The dataset used is a simple sinusoidal function.
* `learned_var`: The goal of the experiment is to spot any similarity between the intrinsic variance of the dataset and the one of the models' predictions. The dataset used is a simple sinusoidal function.
* `quant`: The goal of the experiment is to compare the behavior of 2B2N architectures given different weights quantization techniques. The models are trained in 32-bit precision and evaluated at different bit-widths.
* `tail_<NameDataset>_behavior`: The goal of the experiment is to compare the behavior of B2N architectures on the extremes of the input space.
  The dataset used here consists of samples from a `<NameDataset>` function: a gaussian bell, a sigmoid and a sinusoidal function.
