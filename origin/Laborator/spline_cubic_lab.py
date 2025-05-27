import numpy as np
import matplotlib.pyplot as plt

g = lambda x: np.e**(2*x)

a,b = 0, 1
x_dens = np.linspace(a, b, 500)


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

    for i in range(length_x - 1):
        x0, x1 = X[i], X[i+1]
        y0, y1 = Y[i], Y[i+1]
        h_i = h[i]
        M0, M1 = M[i], M[i+1]

        loc_interval = (x >= x0) & (x <= x1)
        dx = x[loc_interval] - x0
        dx1 = x1 - x[loc_interval]

        term1 = M0 * dx1**3 / (6*h_i)
        term2 = M1 * dx**3 / (6*h_i)
        term3 = (y0 - M0 * h_i **2 / 6) * dx1 / h_i
        term4 = (y1 - M1 * h_i **2 / 6) * dx / h_i

        y_eval[loc_interval] = term1+term2+term3+term4

    return y_eval


for n in range(1, 10):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    X = np.linspace(a, b, n+1)
    y_grafic = g(X)

    spline_y = spline_cubic(X, y_grafic, x_dens)

    ax1.set_title('Aproximare spline cubica')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    ax1.grid(True)
    ax1.plot(x_dens, spline_y, c='r', label="Spline cubic"+str(n))
    
    ax1.plot(x_dens, g(x_dens), c='b', label="f(x)")
    ax1.scatter(X, y_grafic)
    ax1.legend()

    eroare_modul = np.abs(g(x_dens) - spline_y)

    ax2.set_title('Eroare')
    ax2.set_xlabel("x")
    ax2.set_ylabel("Y")
    ax2.plot(x_dens, eroare_modul, 'g--', linewidth=0.7)
    


    plt.tight_layout()
    plt.show()

