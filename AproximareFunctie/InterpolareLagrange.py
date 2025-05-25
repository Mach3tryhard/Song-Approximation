import numpy as np
import matplotlib.pyplot as plt

def interpolare_lagrange(X, Y, x):
    n = len(X)
    y = 0.0
    for i in range(n):
        L_i = 1.0
        for j in range(n):
            if i != j:
                L_i *= (x - X[j]) / (X[i] - X[j])
        y += Y[i] * L_i
    return y

def interpolare_naiva(X, Y, x):
    A = np.vander(X, increasing=True)
    C = np.linalg.solve(A, Y)
    powers = np.arange(len(C))
    return np.sum(C * x**powers)

def grafic_interpolare_valori(X, Y, metoda):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, Y, color="black", s=50, label="Noduri")

    xg = np.linspace(X[0], X[-1], 500)
    
    if metoda.lower() == "lagrange":
        yg = [interpolare_lagrange(X, Y, xi) for xi in xg]
        label = "Lagrange"
        style = 'r-'
    elif metoda.lower() == "naiva":
        yg = [interpolare_naiva(X, Y, xi) for xi in xg]
        label = "Naivă (Vandermonde)"
        style = 'g-'
    else:
        raise ValueError("Metodă necunoscută: use 'lagrange' or 'naiva'")

    ax.plot(xg, yg, style, linewidth=2, label=label)

    ax.set_title(f"Interpolare {label}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    ax.legend()

    plt.tight_layout()
    plt.show()


X = np.linspace(0, 1, 9)
Y = np.array([0.0, 0.5, 0.8, 0.9, 1.0, 0.9, 0.7, 0.4, 0.0])

grafic_interpolare_valori(X, Y, metoda="naiva")

