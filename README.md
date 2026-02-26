Logistic Regression From Scratch

This repository implements a Logistic Regression classifier from scratch in Python, using NumPy (no scikit‑learn model training functions). It trains a model on the Breast Cancer dataset and shows how gradient descent can optimize weights and bias for binary classification.

🚀 Project Overview

This project teaches you how Logistic Regression works under the hood:

Manual implementation of the sigmoid activation

Cost calculation with optional L2 regularization

Gradient descent optimization

Model evaluation and accuracy

Visualization of loss (cost) over iterations

It does not use any machine learning library for the algorithm itself — only NumPy and standard utilities.

📁 Repository Structure
.
├── model.py          # Logistic regression functions (sigmoid, cost, gradient descent)
├── train.py          # Train the model and explore cost history
├── evaluate.py       # Load saved model and test accuracy
├── utils.py          # Helper functions (accuracy, plotting)
├── weights.npy       # Saved model weights (after training)
├── bias.npy          # Saved bias (after training)
├── README.md         # Project documentation

->What’s Inside
->model.py
sigmoid(z): Logistic activation function
compute_cost(...): Calculates binary cross‑entropy loss with regularization
gradient_descent(...): Updates weights and bias using gradient descent
predict(...): Converts probabilities to class labels (0 or 1)

->train.py

This script:

Loads the Breast Cancer dataset

Splits train/test sets

Standardizes features

Runs gradient descent

Plots cost history

Saves weights & bias for later evaluation

->evaluate.py

After training, this script:
Loads saved weights and bias
Makes predictions on train and test data
Computes and prints accuracy scores


->utils.py
Utility functions:
compute_accuracy(y_true, y_pred)
plot_cost(cost_history) for visualizing the training loss

->How to Use

Clone the repo

git clone https://github.com/snippet-com/Logistic-Regression-From-Scratch
cd Logistic-Regression-From-Scratch

Install dependencies

Make sure you have:
Python 3.x
NumPy
Scikit‑learn (for dataset + splitting)
Matplotlib (for plotting)
Train the model
python train.py
Evaluate the trained model
python evaluate.py

Output
Training prints cost every 100 iterations
Plot of cost vs. iteration
Final accuracy on training and test sets
