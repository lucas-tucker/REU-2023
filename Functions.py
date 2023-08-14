import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

# THESE ARE FUNCTIONS IMPORTED INTO OTHER EXPERIMENT FILES

def t_nearest_matr(m, t, X):
    t_nearest = np.ones((m, t), dtype=int) * 1
    for id, row in enumerate(X.T):
        dif = X.T - row # get vector representation-wise differences
        norm_indices = np.argsort(np.linalg.norm(dif, axis = 1))
        t_nearest[id] = norm_indices[1: t + 1]
    return t_nearest # returns m x t matrix representing k_nearest

def weight_matr(m, N, X, sigma):
    X = X.T # transpose the data matrix for ease in W_ij calculation
    W = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i in N[j]:
                W[i][j] = np.exp(np.linalg.norm(X[i] - X[j]) / (sigma ** 2))
    return W

def diag_matr(m, W):
    D = np.zeros((m, m))
    for i in range(m):
        D[i][i] = np.sum(W[i])
    return D

"""
Summary of parameters:
m is the number of points, k is the desired dimension (what we will reduce to), 
t is the number of nearest neighbors, sigma regularizes the weight matrix, and 
X is the n x m data matrix
"""
def get_le_reduced(k, t, sigma, X):
    m = len(X[0])
    N = t_nearest_matr(m, t, X)
    W = weight_matr(m, N, X, sigma)
    D = diag_matr(m, W)
    L = D - W

    print(f"N sample is {N[:6]}")
    print(f"W sample is {W[:6]}")
    print(f"D sample is {D[:6]}")

    eigenvalues, eigenvectors = np.linalg.eig(L)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvectors[-k:] # smallest k eigenvectors (rows)