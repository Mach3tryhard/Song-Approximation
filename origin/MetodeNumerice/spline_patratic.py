import numpy as np

def spline_patratic(X, Y, x_eval):    
    n = len(X) - 1
    h = X[1:] - X[:-1]
    
    c = Y[:-1]
    
    A = np.zeros((2*n, 2*n))
    d = np.zeros(2*n)
    
    for i in range(n):
        A[i, i] = h[i]**2      
        A[i, n + i] = h[i]      
        d[i] = Y[i+1] - c[i]    
    
    for i in range(n - 1):
        A[n + i, i] = 2 * h[i]      
        A[n + i, n + i] = 1          
        A[n + i, n + i + 1] = -1     
        d[n + i] = 0
    
    A[-1, 0] = 1
    d[-1] = 0
    
    coeffs = np.linalg.solve(A, d)
    
    a = coeffs[:n]
    b = coeffs[n:]
    
    y_eval = np.zeros_like(x_eval, dtype=float)
    
    idx = np.searchsorted(X, x_eval, side='right') - 1
    idx = np.clip(idx, 0, n - 1)
    
    dx = x_eval - X[idx]
    
    y_eval = a[idx] * dx**2 + b[idx] * dx + c[idx]
    
    if y_eval.size == 1:
        return y_eval.item()
    return y_eval
