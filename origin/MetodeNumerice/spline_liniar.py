import numpy as np

def spline_liniar(X, Y, x):
    y_val = np.zeros_like(x)
    indices = np.searchsorted(X, x, side='right') - 1
    indices = np.clip(indices, 0, len(X) - 2)
    x0, x1 = X[indices], X[indices+1]  # coefficients
    y0, y1 = Y[indices], Y[indices+1]
    
    y_val = y0 + (y1 - y0) / (x1 - x0) * (x - x0)

    return y_val