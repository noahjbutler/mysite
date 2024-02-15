import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
data = iris.data
labels = iris.target
K = 3


def kmeansclustering(data, K, maxiter=100):
    N, D = data.shape
    r = np.zeros((N, K))
    mu = data[np.random.choice(N, K, replace=False)]
    objective_func_values = []

    for iter in range(maxiter):
        for n in range(N):
            distances = np.sum((data[n, :] - mu)**2, axis=1)
            k = np.argmin(distances)
            r[n, :] = 0
            r[n, k] = 1

        munew = np.zeros((K, D))
        for k in range(K):
            munew[k, :] = np.mean(data[r[:, k] == 1, :], axis=0)

        objectivefuncvalue = np.sum(
            [np.sum((data[r[:, k] == 1, :] - mu[k, :])**2) for k in range(K)])
        objective_func_values.append(objectivefuncvalue)

        if np.allclose(mu, munew, atol=1e-4):
            break

        mu = munew.copy()

    return mu, r, objective_func_values


mu, r, valuesOBJFUNC = kmeansclustering(data, K)
plt.plot(valuesOBJFUNC)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Value vs. Iteration')


def clusters(data, mu, r, K):
    p2 = plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']

    for k in range(K):
        plt.scatter(data[r[:, k] == 1, 0], data[r[:, k] == 1, 1],
                    c=colors[k], label=f'Cluster {k + 1}')

    plt.scatter(mu[:, 0], mu[:, 1], marker='*',
                s=300, c='black', label='Centroids')
    plt.legend()
    plt.title(f'K-means Clustering (K = {K})')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')


def decisionboundary(data, mu, r, K):
    h = 0.02
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x = np.array([xx[i, j], yy[i, j]])
            dist = np.linalg.norm(x - mu[:, :2], axis=1)
            Z[i, j] = np.argmin(dist)
    p3 = plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    markers = ['o', '^', 's']

    for k in range(K):
        plt.scatter(data[r[:, k] == 1, 0], data[r[:, k] == 1, 1],
                    c=colors[k], marker=markers[k], label=f'Cluster {k + 1}')

    plt.contourf(xx, yy, Z, alpha=0.2, colors=colors)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title(f'Decision boundaries for k-means clustering with K={K}')
    plt.legend()
    plt.show()


clusters(data, mu, r, K)
decisionboundary(data, mu, r, K)
