import numpy as np
import matplotlib.pyplot as plt

# RBF Kernel
def rbf_kernel(X, h=-1):
    XY = np.dot(X, X.T)
    X2_ = np.sum(X**2, axis=1).reshape(-1, 1)
    X2 = X2_ + X2_.T - 2*XY
    if h < 0:  # Median heuristic for bandwidth
        h = np.median(X2) / (2 * np.log(X.shape[0] + 1))
    K = np.exp(-X2 / h)
    return K

# Gradient of the RBF Kernel
def grad_rbf_kernel(X, K, h=-1):
    XY = np.dot(X, X.T)
    X2_ = np.sum(X**2, axis=1).reshape(-1, 1)
    X2 = X2_ + X2_.T - 2*XY
    if h < 0:  # Median heuristic for bandwidth
        h = np.median(X2) / (2 * np.log(X.shape[0] + 1))
    dim = X.shape[1]
    dK = np.zeros((X.shape[0], X.shape[0], dim))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dK[i, j, :] = K[i, j] * (X[i, :] - X[j, :]) / h
    return dK

# SVGD Update
def svgd_update(particles, grad_log_p, stepsize=0.1, num_iter=100):
    for _ in range(num_iter):
        # Compute kernel and its gradient
        K = rbf_kernel(particles)
        dK = grad_rbf_kernel(particles, K)

        # Compute SVGD gradient
        grad_logp = grad_log_p(particles)
        phi = np.zeros_like(particles)
        for i in range(particles.shape[0]):
            phi[i, :] = np.sum(K[i, :, np.newaxis] * grad_logp + dK[i, :, :], axis=0)

        # Update particles
        particles += stepsize * phi / particles.shape[0]
    return particles

# Example log probability and gradient (e.g., standard normal distribution)
def log_p(X):
    return -0.5 * np.sum(X**2, axis=1)

def grad_log_p(X):
    return -X

# Initialize particles (n x dim)
n, dim = 10000, 2
particles = np.random.randn(n, dim)

# Visualize initial particles
plt.scatter(particles[:, 0], particles[:, 1], alpha=0.5, label='Initial Distribution')

# Run SVGD
updated_particles = svgd_update(particles, grad_log_p, stepsize=0.1, num_iter=500)

# Visualize updated particles
plt.scatter(updated_particles[:, 0], updated_particles[:, 1], alpha=0.5, label='Updated Distribution')
plt.legend()
plt.title("SVGD Sampling")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig('comparison.png')
