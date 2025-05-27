import numpy as np
import matplotlib.pyplot as plt

g = lambda x: np.e**(2*x)
a,b = 0, 1
x_dens = np.linspace(a, b, 500)


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
    for i in range(nr_int):
        x0 = X[i]
        loc_interval = (x >= X[i]) & (x <= X[i+1])
        dx = x[loc_interval] - x0
        y_eval[loc_interval] = a[i]*dx**2 + b[i]*dx +c[i]

    return y_eval


for n in range(1, 4):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    X = np.linspace(a, b, n+1)
    y_grafic = g(X)

    spline_y = spline_patratic(X, y_grafic, x_dens)
    
    ax1.set_title('Aproximare spline liniara')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    ax1.grid(True)
    ax1.plot(x_dens, spline_y, c='r', label="Spline patratic "+str(n))
    
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

