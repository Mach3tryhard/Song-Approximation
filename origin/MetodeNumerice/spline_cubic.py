import numpy as np

def spline_cubic(X, Y, x):
    length_x = len(X)
    h = X[1:] - X[:-1] 

    #rezolvare sistem de derivate de ord 2
    A = np.zeros((length_x,length_x))
    vect_b = np.zeros(length_x)

    A[0,0]=1
    A[-1,-1]=1

    for i in range(1, length_x - 1):
        A[i, i -1] = h[i-1]
        A[i,i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        vect_b[i] = 6 * ((Y[i+1] - Y[i]) / h[i] - (Y[i] - Y[i-1]) / h[i-1])
    
    M = np.linalg.solve(A, vect_b) #TO DO: REZOLVARE CA LA LAB!!!!!

    y_eval = np.zeros_like(x)

    idx = np.searchsorted(X, x) - 1 # tot indexing optimizat
    idx = np.clip(idx, 0, length_x - 2)

    x0 = X[idx]
    x1 = X[idx + 1]
    y0 = Y[idx]
    y1 = Y[idx + 1]
    h_i = x1 - x0
    M0 = M[idx]
    M1 = M[idx + 1]

    dx = x - x0
    dx1 = x1 - x

    term1 = M0 * dx1**3 / (6 * h_i)
    term2 = M1 * dx**3 / (6 * h_i)
    term3 = (y0 - M0 * h_i**2 / 6) * dx1 / h_i
    term4 = (y1 - M1 * h_i**2 / 6) * dx / h_i

    y_eval = term1 + term2 + term3 + term4
    return y_eval
