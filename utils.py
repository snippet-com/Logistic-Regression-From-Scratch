import matplotlib.pyplot as plt
import numpy as np

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def plot_cost(cost_history):
    import matplotlib.pyplot as plt
    plt.plot(cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.show()