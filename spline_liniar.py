import numpy as np
import matplotlib.pyplot as plt


g = lambda x: np.e**(2*x)
a,b = 0, 1


x_dens =  np.linspace(a, b, 500) # multe valori x din [a, b] pt a evalua spline-ul

def spline_liniar(X, Y, x): # x e un array
    y_val = np.zeros_like(x)
    # Pasul 1: Determinam coeficientii polinoamelor liniare pe fiecare subinterval
    for i in range(len(X) - 1):
        x0, x1 = X[i], X[i+1] #coeficienti poly
        y0, y1 = Y[i], Y[i+1]

        loc_interval = (x >= x0) & (x <= x1)
        #(x >= x0) returneaza un array cu elemente True unde x[] >= x0, la fel si celalalt dar <=. Operatorul AND lasa doar o valoare
        
        y_val[loc_interval] = y0 + (y1 - y0) / (x1 - x0) * (x[loc_interval] - x0)

    return y_val
        
for n in range(1, 4):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    X = np.linspace(a, b, n+1)
    y_grafic = g(X)

    spline_y = spline_liniar(X, y_grafic, x_dens)

    ax1.set_title('Aproximare spline liniara')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    
    ax1.grid(True)
    ax1.plot(x_dens, spline_y, c='r', label="Spline liniar "+str(n))
    ax1.plot(x_dens, g(x_dens), c='b', label="f(x)")
    ax1.scatter(X, y_grafic)
    ax1.legend()

    eroare_modul = np.abs(g(x_dens) - spline_y)

    ax2.set_title('Eroare')
    ax2.set_xlabel("x")
    ax2.set_ylabel("Y")
    ax2.plot(x_dens, eroare_modul, 'g--', linewidth=0.7)
    ax2.legend()


    plt.tight_layout()
    plt.show()



