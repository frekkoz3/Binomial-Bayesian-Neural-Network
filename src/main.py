r"""
██████  ██████  ██████  ███    ██ 
     ██ ██   ██      ██ ████   ██ 
 █████  ██████   █████  ██ ██  ██ 
██      ██   ██ ██      ██  ██ ██ 
███████ ██████  ███████ ██   ████                                 
"""

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    # x-axis
    x = torch.linspace(-5, 5, 1000)

    # different standard deviations
    sigmas = [0.01, 0.1, 0.5, 1.0, 2.0]

    plt.figure(figsize=(8, 5))

    for sigma in sigmas:
        dist = torch.distributions.Normal(loc=0.0, scale=sigma)
        cdf = dist.cdf(x)
        plt.plot(x.numpy(), cdf.numpy(), label=f"sigma^2 = {sigma**2}")

    plt.title("Gaussian CDF for different variances (mu = 0)")
    plt.xlabel("x")
    plt.ylabel("CDF Φ(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

    