# Binomial Bayesian Neural Network

a project by Team Rocket - Bredariol, Riccio, Savorgnan

---

## The Idea

A **Binomial Bayesian Neural Network (2B2N)** is a Bayesian Neural Network whose weights are parameterized through a remapped Binomial distribution with a fixed number of trials (N).

At first glance, 2B2N may appear to be a purely experimental architecture. However, the original motivation behind this project was much more practical. A Binomial distribution is completely characterized by a single parameter (p \in [0,1]) when (N) is fixed. Consequently, each weight distribution can be represented using only this parameter, potentially reducing the memory footprint of Bayesian neural networks significantly. By storing (p) in a compact numerical format, this reduction can be achieved natively rather than through post-training quantization techniques.

What began as an exploration of memory-efficient Bayesian models soon evolved into a broader research interest. The simplicity of the underlying distribution raises several intriguing questions: how to perform efficient inference, how to interpret the learned uncertainty, how the expressiveness of the model compares to traditional Gaussian-based BNNs, and what advantages or limitations emerge from such a constrained probabilistic representation.

The 2B2N project is therefore both an investigation into a novel Bayesian neural network architecture and an exploration of the trade-offs between representational simplicity, memory efficiency, and predictive uncertainty.

## Repository Structure

## Formulation

## Quick Setup
