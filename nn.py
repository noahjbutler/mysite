import csv
import numpy as np
import matplotlib.pyplot as plt


def sig(x):
    return 1 / (1 + np.exp(-x))


with open('iris.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    data = []
    for row in readCSV:
        if len(row) >= 5 and row[4] != 'species':
            data.append(row[:4])

data = np.array(data, dtype=float)
X = np.vstack((np.hstack((np.ones((50, 1)), data[50:100, :3])), np.hstack(
    (np.ones((50, 1)), data[100:150, :3]))))
y = np.hstack((np.ones(50), 2 * np.ones(50)))


def mse(X, W, y):
    a = np.dot(X, W)
    y_pred = sig(a)
    error = y_pred - (y == 2)
    return np.mean(error ** 2)


def gradi(X, W, y):
    a = np.dot(X, W)
    y_pred = sig(a)
    dy = y_pred - (y == 2)
    da = y_pred * (1 - y_pred)
    dW = X.T
    return np.dot(dW, dy * da)


W = np.zeros(4)
alpha = 0.1

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 1], X[y == 1, 2], c='b', label='Class 1')
plt.scatter(X[y == 2, 1], X[y == 2, 2], c='r', label='Class 2')
W_init = np.array([-5, 1, 1])
plt.plot([-2, 5], [(-W_init[0] - W_init[1] * (-2)) / W_init[2],
         (-W_init[0] - W_init[1] * 5) / W_init[2]], 'm', label='Initial Decision Boundary')

for i in range(10):
    idx = np.random.randint(len(X))
    x = X[idx]
    y_i = y[idx]
    grad = gradi(x, W, y_i)
    W -= alpha * grad

    plt.plot([-2, 5], [(-W[0] - W[1] * (-2)) / W[2],
             (-W[0] - W[1] * 5) / W[2]], label=f'Decision Boundary {i+1}')

plt.legend()
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Iris Dataset - Decision Boundary with Small Steps')
plt.show()
