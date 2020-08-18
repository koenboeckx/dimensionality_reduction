"""Implement unit test for SNE:
(1) Generate 3D data matrix of randomly sampled points on two different 2D-manifolds (e.g. planes)
(2) Observe (and potentially correct) behavior of SNE
"""

import numpy as np

from sne import SNE, plot_Y
 
N = 150
x, y = np.random.rand(N), np.random.rand(N)
z1 = 3*x[:50] + 4*y[:50] + 10
z2 = 3*x[50:100] + 4*y[50:100] + 20
z3 = 4*x[100:] + 4*y[100] + 20
z  = np.hstack([z1, z2, z3])
X = np.vstack([x, y, z]).T
labels = [1,]*50 + [2,]*50 + [3,]*50

Y = SNE(X)
plot_Y(Y, labels)


print('.................')