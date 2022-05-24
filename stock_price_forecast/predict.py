import numpy as np
import matplotlib.pyplot as plt


def predict(x_test, y_test, model, indeces):

    pred = model.predict(x_test)
    error = {}

    prices = ["Open", "High", "Low", "Close"]
    n = x_test.shape[0] + x_test.shape[1]
    fig, axes = plt.subplots(x_test.shape[0], 1, figsize=(18, 1 *x_test.shape[0]))
    for i, price in zip(range(y_test.shape[2]), prices):
        error[price] = np.sum((y_test[:, :, i] - pred[:, :, i])**2) / n
        for j in range(x_test.shape[0]):
            axes[j].plot(indeces[j, :], y_test[j, :])
            axes[j].plot(indeces[j, :], pred[j, :])

    return error