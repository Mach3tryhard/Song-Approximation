import numpy as np

def spline_patratic(X, Y, x):
    nr_int = len(X) - 1
    a = np.zeros(nr_int)
    b = np.zeros(nr_int)
    c = Y[:-1] 

    h = X[1:] - X[:-1] 

    b[0] = (Y[1] - Y[0]) / h[0] # primu slope initial
    a[0] = 0 #prima derivata egal 0

    for i in range(1, nr_int):
        b[i] = 2*(Y[i+1] - Y[i])/h[i] - b[i-1]
        a[i] = (Y[i+1] - Y[i] - b[i]*h[i]) / (h[i]**2)
    
    y_eval = np.zeros_like(x)

    idx = np.searchsorted(X, x) -1 # idx e indexul intervalului lui x curent
    idx = np.clip(idx, 0, nr_int - 1)

    dx = x - X[idx]
    y_eval = a[idx] * dx ** 2 + b[idx] * dx + c[idx]

    return y_eval
