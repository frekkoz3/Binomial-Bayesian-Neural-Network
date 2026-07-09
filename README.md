# Binomial Bayesian Neural Network

a project by Team Rocket - Bredariol, Riccio, Savorgnan

---

## The Idea

A **Binomial Bayesian Neural Network (2B2N)** is a Bayesian Neural Network whose weights are parameterized through a remapped Binomial distribution with a fixed number of trials (N).

At first glance, 2B2N may appear to be a purely experimental architecture. However, the original motivation behind this project was much more practical. A Binomial distribution is completely characterized by a single parameter (p \in [0,1]) when (N) is fixed. Consequently, each weight distribution can be represented using only this parameter, potentially reducing the memory footprint of Bayesian neural networks significantly. By storing (p) in a compact numerical format, this reduction can be achieved natively rather than through post-training quantization techniques.

What began as an exploration of memory-efficient Bayesian models soon evolved into a broader research interest. The simplicity of the underlying distribution raises several intriguing questions: how to perform efficient inference, how to interpret the learned uncertainty, how the expressiveness of the model compares to traditional Gaussian-based BNNs, and what advantages or limitations emerge from such a constrained probabilistic representation.

The 2B2N project is therefore both an investigation into a novel Bayesian neural network architecture and an exploration of the trade-offs between representational simplicity, memory efficiency, and predictive uncertainty.

## Repository Structure
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

## Quick Setup
Just install requirements and go!

```bash
pip install -r requirements.txt
```

To run a training session, simply adapt the configuration directly in `src/train/train.py` and execute the following command from terminal:

```bash
python -m src.train.train
```