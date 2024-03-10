


from typing import Tuple

import numpy as np



def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    rows, cols = 480, 640
    gradient = np.linspace(0, 1, rows)[:, None] * np.ones((1, cols))
    res[0:480, 0:640] = gradient

    rows, cols = 240, 320
    x = np.linspace(1, -0.1, cols)
    y = np.linspace(-0.1, 1, rows)
    # create a 2D grid of x and y values
    x, y = np.meshgrid(x, y)
    # calculate the gradient using the Euclidean distance formula
    gradient2 = -np.sqrt(x**2 + y**2)
    res[240:480, 320:640] = gradient2

    x_2 = np.linspace(-0.1, 1, cols)
    y_2 = np.linspace(-0.1, 1, rows)
    x_2, y_2 = np.meshgrid(x_2, y_2)
    gradient3 = np.sqrt(x_2**2 + y_2**2)
    res[240:480, 0:320] = gradient3
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    rows, cols = 480, 640
    gradient = np.linspace(0, 1, rows)[:, None] * np.ones((1, cols))
    res[0:480, 0:640] = gradient
    rows, cols = 240, 320
    x = np.linspace(-0.1, 1, cols)
    y = np.linspace(-0.1, 1, rows)

    # create a 2D grid of x and y values
    x, y = np.meshgrid(x, y)

    # calculate the gradient using the Euclidean distance formula
    gradient2 = -np.sqrt(x**2 + y**2)

    res[240:480, 0:320] = gradient2

    x_2 = np.linspace(1, -0.1, cols)
    y_2 = np.linspace(-0.1, 1, rows)
    x_2, y_2 = np.meshgrid(x_2, y_2)
    gradient3 = np.sqrt(x_2**2 + y_2**2)
    res[240:480, 320:640] = gradient3
    return res
