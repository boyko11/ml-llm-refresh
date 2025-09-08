import numpy as np


def mse_loss(y_predicted, y_actual):

    return 0.5 * np.square(y_predicted - y_actual).mean()

y_predicted = np.array([1, 2, 3, 4, 5])
y_actual = [5, 4, 3, 2, 1]

print(mse_loss(y_predicted, y_actual))