# Binomial Bayesian Neural Network

This file describes in detail the theory behind our work.
Most of the details are present in referenced material.

## Contents

<!-- TOC -->
* [Binomial Bayesian Neural Network](#binomial-bayesian-neural-network)
  * [Contents](#contents)
  * [Binomial Weights in Neural Networks](#binomial-weights-in-neural-networks)
  * [Gaussian Approximation](#gaussian-approximation)
  * [Gumbel-Softmax Reparameterization Trick](#gumbel-softmax-reparameterization-trick)
  * [Weight Initialization](#weights-initialization)
* [References](#references)
<!-- TOC -->

## Binomial Weights in Neural Networks

The core idea behind our project is to analyze the behaviour of Neural Networks in the case where weights are discrete and evenly spaced among an interval $[v_{min}, v_{max}]$.  
This approach can be useful in cases where high precision and high throughput are coupled with memory constraints.

It can also be interesting to introduce a prior distribution on the weights, so to apply the well-known Bayesian framework and to obtain a flexible architecture. 
In our case, we choose to model each weight $w^{(i)}$ as a binomial random variable:

$$
  \tilde w^{(i)} \sim \text{ Bin}(N, p^{(i)})
$$

where $N$ is a fixed integer parameter indicating the number of possible weight outcomes, i.e., the support of the binomial distribution $\{0, 1,\dots, N\}$, and $p^{(i)} $ the *learned* probability of success for the $i$-th weight.  
The "partial" weights are then remapped in the chosen interval by using the affine transformation:

$$
  w^{(i)} = v_{min} + \tilde w^{(i)} \cdot \frac{v_{max} - v_{min}}{N}
$$

In order to learn a valid probability $p^{(i)}$, we introduce a learnable parameter $\rho^{(i)}$ and we apply the logistic function to it:

$$
    p^{(i)} = \sigma(\rho^{(i)}) = \frac{1}{1 + e^{-\rho^{(i)}}}
$$

The current setup appears to be a good compromise between the flexibility of the model and the number of learnable parameters, which is kept low. However, it still lacks a crucial aspect: the possibility to compute gradients and perform backpropagation. This is a well-known problem for discrete neural networks; in literature several approaches have been followed to partially sacrifice the discreteness of the model but to allow gradients to flow (and thus, to allow the models to be trained by using standard gradient descent methods). We have implemented in our framework two solutions, here presented: the Gaussian approximation and a generalized version of the Gumbel-Softmax reparameterization trick.

## Gaussian Approximation

The Gaussian approximation comes straightforward from the Central Limit Theorem (CLT): for sufficiently large values of $N$ the Binomial distribution $w \sim \text{Binomial}(N, p)$ can be approximated as a gaussian $\mathcal{N}(\mu = Np, \sigma^2 = Np(1-p))$.  
Thus, we can simply apply the reparameterization trick as presented in [1](#kinga2014):

$$
  \tilde w \approx Np + \sqrt{Np(1-p)} \cdot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,1).
$$

The simple rounding function $\text{ round()}$ is not sufficient in the sense that it introduces a non-differentiable step; we can go around this problem exploiting the Straight-Through Estimator (STE) trick:

$$
w = \tilde w + \text{SG}\big( \text{ round}(\tilde w) - \tilde w \big)
$$

where $\text{SG}$ is a stop-gradient step that ensures full differentiability: it preserves the rounded value during the forward pass, while propagating the identity gradient during the backward.

After this step, the obtained binomial weight is mapped into its valid interval $[v_{min}, v_{max}]$ using the already presented formula.

## Gumbel-Softmax Reparameterization Trick

The Gumbel-Softmax reparameterization trick (GS) is a well-known technique for training discrete neural networks where weights come from a categorical distribution [2](#jang2016), [3](maddison2016).  
The idea behind GS is to rewrite the categorical distribution as a Gumbel, and approximate the sampling from the Gumbel, which is usually done with a non-differentiable $\arg \max$ step, by using a differentiable and parametrizable $ \text{ softmax}$ function.

Suppose to have samples from a categorical distribution

$$
  c_1, \dots, c_n \sim \text{ Cat}(0, N)
$$

having class probabilities $\pi = \{\pi_0, \pi_1, \dots, \pi_N \} $. For practicality, we suppose that each sample $c_j$ is one-hot encoded, so that (when transposed) looks like:

$$
  c_j^T = [0_0 \quad 0_1 \quad \dots \quad 1_j \quad \dots \quad 0_N]
$$

The reparameterization trick works as follows.

1. Sample $N+1$ parameters $u_0, \dots, u_N \sim \text{ Uniform}(0, 1) $, one for each value supported by $\text{Cat}$.

2. Generate for each $u_i$ an auxiliar sample from a Gumbel distribution, as:$$g_i = - \log(-\log(u_i))$$

3. Apply the softmax function to $g_i + \log \pi_i$:$$
    s_i = \text{ softmax}(g_i + \log(\pi_i)) = \frac{\exp\big( (g_i + \log \pi_i) / \tau \big)}{\sum_{j=0}^N \exp\big((g_j + \log \pi_j)/\tau\big)}$$
  where $\tau$ is the temperature parameter that, in practice, is annealed during learning.  
  When $\tau$ is small enough, the target one-hot $c_i$ is recovered; conversely, for $\tau \rightarrow \infty$ the distribution tends to a discrete uniform over the same support $\{ 0, \dots, N \}$.

4. When dealing with neural network weights, we need to remap the obtained vector to a single scalar $\tilde w_j$, so that it now belongs to the interval defined by the extremes of the categorical distribution $ [0, N] $:$$
  \tilde w_j \approx \sum_{i=0}^N s_i \cdot i$$

The approximation, actually, arises from the softmax step, which approximates the discrete probability by smoothing it in the continuous space.  
Notice how, after this scheme, the distribution becomes fully differentiable.

The last aspect to consider is that the resulting weights $\tilde w_i$ now belong to a continuous space, so we need to remap them into our discrete support $\{0, \dots, N\} $.
As described previously, we can once again apply the STE trick.

This GS reparameterization trick does not natively apply to the binomial distribution.  
However, we can think of the binomial as a categorical distribution where each probability $\pi_i$ is taken from the binomial probability mass function:

$$
  \pi_i = {N \choose i} p^i (1-p)^{N-i}
$$

[4](#joo2025) proved that this approach works for each discrete distribution with finite support, as the binomial.  
Thus, we can apply the previous scheme in our setup.

## Weights Initialization

The initialization of the weights is a crucial step in the training of neural networks.  
In our case, we saw that our model suffers from very high instability if weights are initialized too widely, i.e., if the variance of the $\rho$ parameter, $\sigma_\rho^2$, is too wide.

Let's provide a bit of calculus.  
We suppose the following:

$$
  \begin{align}
    \omega &= v_{min} + \frac{(v_{max} - v_{min})}{N} \cdot \tilde \omega  \\
            &:= v_{min} + \frac{\Delta}{N} \cdot \tilde \omega  \\
            &:= v_{min} + S \cdot \tilde \omega \\
    \tilde \omega &\sim \text{Binomial}(N, p) \\
    p &= \text{sigmoid}(\rho) \\
    \rho &\sim \mathcal{N}(0, \sigma^2_\rho) \\
    x &\sim \mathcal{N}(0, 1)
  \end{align}
$$

where $x$ are our data, and $\omega$ the weights of the network; other parameters, $v_{min}, v_{max}, N$ are fixed.  
Suppose also to have a naïve MLP, composed by a single layer with just one neuron $y$, whose activation function is the identity, thus there is no non-linearity (applying a ReLU should just halves the variance). The input dimension is $D$. The results we obtain will be generalizable to any neuron of any layer at any depth $t$; in that case, $D$ will be the dimension of the layer at depth $t-1$. Finally, we suppose that weights and data inputs are independent (this is false as the training goes on but it is considerably true in the beginning).

Assuming $\mathbb{x} \sim \mathcal{N}(0, 1)$, the variance of the single neuron will be:

$$
\begin{align}
  \mathbb{V}[y] &= \mathbb{V}[\sum_i^D \omega_i x_i] \\
  &= \sum_i^D \mathbb{V}[\omega_i x_i] \\
  &= D \mathbb{V}[\omega]\mathbb{V}[\mathbb{x}] \\
  &= D \mathbb{V}[\omega]
\end{align}
$$

The equation above holds because of the symmetry around 0. Indeed:
$$
\begin{align}
  \mathbb{V}[\omega x] &= \mathbb{V}[\omega]\mathbb{V}[x] + \mathbb{V}[\omega](\mathbb{E}[x])^2 + \mathbb{V}[x](\mathbb{E}[\omega]) \\
                        &= \mathbb{V}[\omega]\mathbb{V}[x] + 0 + \mathbb{E}[\omega] \\
\end{align}
$$
By applying simple calculus, we have the following:
$$
\begin{align}
  \mathbb{E}[\omega] &= \mathbb{E}[v_{\min} +S \cdot \tilde \omega] \\
        &= \mathbb{E}[\tilde \omega]S + v_{\min} \\
  \mathbb{V}[\omega] &= \mathbb{V}[v_{\min} + S \cdot \tilde \omega] \\
        &= \mathbb{V}[\tilde \omega]S^2
\end{align}
$$

Now, let's prove that $\mathbb{E}[\tilde \omega] = N \mathbb{E}[\sigma(\rho)] = N/2$.  
At first, recall that since the sigmoid function $\sigma$ is symmetric under rotations, it holds that:

$$
  \sigma(-x) = 1 - \sigma(x) 
$$

and that the gaussian function is an odd function, thus $f(x) = f(-x)$.  
So:

$$
\begin{align}
\mathbb{E}[\sigma(\rho)] &= \int_{-\infty}^{+\infty} \sigma(x) f(x)dx \\
  &= \int_{+\infty}^{-\infty}\sigma(-x) f(-x) d(-x) \\ 
  &= \int_{-\infty}^{+\infty} \sigma(-x) f(-x) d(x) \\
  &= \int_{-\infty}^{+\infty} \big(1-\sigma(x)\big) f(x) d(x) \\
  &= \int_{-\infty}^{+\infty} f(x) d(x) - \int_{-\infty}^{+\infty} \sigma(x) f(x) d(x) \\
  &= 1 - \mathbb{E}[\sigma(\rho)] \\ 
2\mathbb{E}[\sigma(\rho)] &= 1 \\
\mathbb{E}[\sigma(\rho)] &= 1/2
\end{align}
$$

So that, consequently:
$$
  \mathbb{E}[\tilde \omega] = N \cdot 1/2 = \frac{N}{2}
$$

Now consider $\mathbb{V}[\tilde \omega]$. By applying the law of the total variance we get:

$$
\begin{align}
  \mathbb{V}[\tilde \omega] &= \mathbb{V}[\mathbb{E}[\tilde \omega | \rho]] + \mathbb{E}[\mathbb{V}[\tilde \omega | \rho]] \\
  &= \mathbb{V}[N \sigma(\rho)] + \mathbb{E}[N \sigma(\rho)\big(1-\sigma(\rho)\big)] \\
  &= N^2\mathbb{V}[\sigma(\rho)] + N \mathbb{E}[\sigma(\rho) - \sigma^2(\rho)] \\
  &= N^2 \big(\mathbb{E}[\sigma^2(\rho)] - \mathbb{E}[\sigma(\rho)]^2 \big) + N \mathbb{E}[\sigma(\rho)] - N \mathbb{E}[\sigma(\rho)^2] \\ 
  &= N^2 \mathbb{E}[\sigma(\rho)^2] - N^2 \cdot (\frac{1}{2})^2 + N \cdot \frac{1}{2} - N \mathbb{E}[\sigma(\rho)^2] \\
  &= \mathbb{E}[\sigma(\rho)^2] (N^2 - N) + \frac{N}{2} - \frac{N^2}{4}
\end{align}
$$

Let'a call, for the seek of keeping the notation simple, $\mathbb{E}[\sigma(\rho)^2] = m_2$, since it is the second moment of the sigmoid function, which is unknown due to the inner dependency on the gaussian distribution (this is actually called the logistic-normal).  
Moreover, notice that the variance has a linear form, thus we will set $M = N(N-1)$ and $Q = \frac{N}{2} - \frac{N^2}{4}$ following the old reminiscences from lyceums ($y = mx + q$).

Putting things together, we get that:

$$
\mathbb{V}[y] = D \cdot S^2 \cdot (M m_2 + Q)
$$

We want to have a bounded (or we can fix) the variance of the activation to a small constant $\varepsilon$.  
So:

$$
\begin{align}
  \varepsilon &= \mathbb{V}[y] \\
  &= D \cdot S^2 \cdot (M m_2 + Q) \\
  m_2 &= \frac{\varepsilon - D \cdot S^2 \cdot Q}{D \cdot S^2 \cdot M} \\
      &:= Y
\end{align}
$$

We want to manipulate this result to explicit $\sigma^2_\rho$ (do **not** confuse the two sigmas!).
At first, remember that the derivative of the sigmoid function is $\sigma'(x) = \frac{\exp(-x)}{(1+\exp(-x))^2}$.
If $\sigma^2_\rho$ is small enough, close to 0, we can apply the Delta method to the sigmoid function $\sigma(\rho)$:

$$
\begin{align}
  \sigma(\rho) &\approx \sigma(0) + \sigma'(0)\rho + \mathcal{o}(\rho^2) \\ 
  \mathbb{V}[\sigma(\rho)] &\approx \mathbb{V}[\sigma(0) + \sigma'(0) \rho] \\
              &= \frac{1}{16} \cdot \mathbb{V}[\rho] \\ 
              &= \frac{1}{16} \sigma^2_\rho \\
  m_2 - \mathbb{E}[\sigma(\rho)]^2 &= \frac{1}{16} \sigma^2_\rho \\ 
  m_2 - \frac{1}{4} &= \frac{1}{16} \sigma^2_\rho \\
  m_2 &= \frac{\sigma^2_\rho + 4}{16}
\end{align}
$$

So that

$$
\begin{align}
  \frac{\sigma^2_\rho + 4}{16} &= Y \\
  \sigma^2_\rho &= 16 Y - 4
\end{align}
$$

Notably, we need to ensure that the resulting quantity is defined positive ($16Y > 4$).  
here things start to behave messily. Indeed we get:

$$
\begin{align}
  \varepsilon - D \cdot S^2 \cdot Q &> \frac{1}{4} \cdot D \cdot S^2 \cdot M \\
  4\varepsilon - 4 \cdot D \cdot S^2 \cdot Q &> D \cdot S^2 \cdot M \\
  D (4 S^2 Q + S^2M) &< 4\varepsilon \\
  D &< \frac{4\varepsilon}{S^2 (4Q + M)} \\
  D &< \frac{4\varepsilon}{\frac{\Delta^2}{N^2}(4Q + M)} \\
  D &< \frac{4\varepsilon}{\frac{\Delta^2}{N^2}(2N - N^2 + N^2 - N)} \\ 
  D &< \frac{4\varepsilon}{\frac{\Delta^2}{N^2}N} \\
  D &< \frac{4N \varepsilon}{\Delta^2}
\end{align}
$$

That is, our initialization scheme works properly only if the number of neurons in a layer scales proportionally with the input size.  
In deep networks, this becomes simply untractable.  
Moreover, if we set $D > N$, the scheme returns a negative variance, which is impossible: networks *cannot* shrink.

Let's now reason on the same result but from a different perspective.  
Since all terms are positive, we can just reformulate the inequality as:

$$
\begin{align}
    \Delta^2 &< \frac{4N \varepsilon}{D} \\
    \Delta &< 2\sqrt{\frac{N\varepsilon}{D}}
\end{align}
$$

From this perspective, we are keeping fixed the dimensions of the input and of the layer: what needs to be adapted is the width of the interval, which now needs to be computed layer-wisely.



## References

1. <a id="kinga2014"></a> Kingma, D., Welling, M., 2014, "Auto-Encoding Variational Bayes."
2. <a id="jang2016"></a> Jang, E., et al., 2016, "Categorical reparameterization with gumbel-softmax."
3. <a id="maddison2016"></a> Maddison, C., et al., 2016, "The concrete distribution: A continuous relaxation of discrete random variables."
4. <a id="joo2025"></a> Joo, W., et al., 2025, "Generalized Gumbel-Softmax gradient estimator for generic discrete random variables."
