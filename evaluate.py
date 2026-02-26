import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from model import predict
from utils import compute_accuracy

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split (same as train.py)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load trained parameters
w = np.load("weights.npy")
b = np.load("bias.npy")

# Predict
y_pred_train = predict(X_train, w, b)
y_pred_test = predict(X_test, w, b)

# Accuracy
print("Training Accuracy:", compute_accuracy(y_train, y_pred_train))
print("Test Accuracy:", compute_accuracy(y_test, y_pred_test))