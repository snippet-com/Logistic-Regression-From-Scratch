import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import gradient_descent
from utils import plot_cost

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize parameters
w = np.zeros(X_train.shape[1])
b = 0
alpha = 0.01
iterations = 1000
lambda_ = 0.1

# Train model
w, b, cost_history = gradient_descent(X_train, y_train, w, b, alpha, iterations, lambda_)

# Optional: plot cost
plot_cost(cost_history)

# Save model parameters (optional)
np.save("weights.npy", w)
np.save("bias.npy", b)