import numpy as np


def relative_error(origin, prediction):
    mask = (origin == 0.0)
    eps = 1e-7
    ##Prevent division by 0.
    prediction[mask] += eps
    origin[mask] += eps
    return np.abs((prediction - origin) / origin)
