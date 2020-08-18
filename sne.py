# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
"""Stochastic Neighbor Embedding"""
import numpy as np 
from matplotlib import pyplot as plt 

from mnist.tools import *

def compute_P(X, goal_perp=10):
    """Compute similarity matrix P. Uses binary search such that
    perplexity of each row is approximately (+/- 1) equal to goal_perp"""
    N, m = X.shape
    P = np.zeros((N, N))
    sigmas = np.ones(N)*50
    for i in range(N):
        delta = 50
        while True:
            for j in range(N):
                if i != j:
                    P[i,j] = np.exp(1/m*np.sum((X[i] - X[j])**2)/(2*sigmas[i]))
            # normalize
            P[i, :] = P[i, :] / np.sum(P[i, :] )
            perp = compute_perplexity(P[i,:])
            if np.abs(perp - goal_perp) < 1:
                break
            elif perp  > goal_perp:
                sigmas[i] = sigmas[i] - delta/2
            else:
                sigmas[i] = sigmas[i] + delta/2
            delta /= 2
    return P, sigmas


def compute_perplexity(p):
    "Computes the perplexity 2**H of a distriubtion p"
    N = len(p)
    x = p * np.log2(p)
    x = x[~np.isnan(x)]  # filter out nan
    H = -np.sum(x)
    return 2**H

def compute_Q(Y):
    """Compute the similarity matrix Q in low-dimension"""
    N, _ = Y.shape
    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                Q[i,j] = np.exp(np.sum((Y[i] - Y[j])**2))
    # normalize rows
    Q /= np.sum(Q, axis=0)[:,None]
    return Q

def compute_grads(Y, P, Q):
    "Computes the gradient matrix dC/dy_i"
    N, m = Y.shape
    grads = np.zeros((N, 2))
    for i in range(N):
        for j in range(N):
            grads[i, :] += 2*(P[i, j] - Q[i, j] + P[j, i] - Q[j, i])*(Y[i,:]-Y[j,:])
    return grads
 
def SNE(X, zeta=0.001, alpha=0.1, eps=5e-3, alpha_decay_rate=0.999):
    N, m = X.shape
    P, _ = compute_P(X)
    Y = np.random.rand(N, 2)
    Y_prev = np.zeros_like(Y)

    diff, ii = 10, 0
    while diff > eps:
        # 3. Compute Q
        Q = compute_Q(Y)
        # 4. Compute gradients
        grads = compute_grads(Y, P, Q)
        # 5. Perform gradient descent step
        diff = np.linalg.norm(Y - Y_prev)
        if ii > 0 and ii % 10 == 0:
            print(f"{ii}: {diff:9.8f}, alpha={alpha:5.4f}")
        Y_new = Y + zeta * grads + alpha * (Y - Y_prev)
        Y_prev = Y
        Y = Y_new
        ii += 1
        alpha *= alpha_decay_rate
    return Y

def plot_Y(Y, labels):
    plt.figure(figsize=(12, 12))
    for label in set(labels):
        idxs = np.where(np.array(labels) == label)[0]
        plt.scatter(Y[idxs, 0], Y[idxs, 1], label=label)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # load data
    N = 100
    images, labels = load_data()
    images, labels = images[:N], labels[:N]
    Y = SNE(images)

    plot_Y(Y, labels)