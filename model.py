import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b, lambda_):
    m = X.shape[0]
    z = np.dot(X, w) + b
    h = sigmoid(z)
    epsilon = 1e-8
    cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    reg_term = (lambda_ / (2*m)) * np.sum(w**2)
    return cost + reg_term

def gradient_descent(X, y, w, b, alpha, iterations, lambda_):
    m = X.shape[0]
    cost_history = []
    for i in range(iterations):
        z = np.dot(X, w) + b
        h = sigmoid(z)
        dw = (1/m) * np.dot(X.T, (h - y)) + (lambda_/m) * w
        db = (1/m) * np.sum(h - y)
        w -= alpha * dw
        b -= alpha * db
        cost_history.append(compute_cost(X, y, w, b, lambda_))
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost_history[-1]:.4f}")
    return w, b, cost_history

def predict(X, w, b):
    z = np.dot(X, w) + b
    return (sigmoid(z) >= 0.5).astype(int)