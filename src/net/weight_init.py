import numpy as np

def weight_initialization (n, n_i, max_val, min_val):

    r = n / 2 - n**2 / 4
    k = (max_val - min_val) / n
    y = (1 - n_i * r * (k ** 2)) / (n_i * (k**2) * n * (n-1))

    sigma = 16 * y - 4

    return sigma



if __name__ == "__main__":
    n = 20
    n_i = 10

    min_val = -2
    max_val = 2

    print(weight_initialization(n, n_i, max_val, min_val))

    sigmas = [[weight_initialization(n, n_i, max_val, min_val) for n_i in range(10, 100, 5)] for n in range(10, 100, 5)]

    # Plot sigmas

    import matplotlib.pyplot as plt
    x = [n for n in range(10, 100, 5)]
    y = [n_i for n_i in range(10, 100, 5)]
    # print 3D

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    Z = np.array(sigmas)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('n')
    ax.set_ylabel('n_i')
    ax.set_zlabel('sigma')
    plt.show()
