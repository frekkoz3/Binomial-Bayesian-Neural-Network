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
