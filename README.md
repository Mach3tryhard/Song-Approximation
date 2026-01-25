# Aproximarea unei melodii folosind mai multe metode de calcul numeric

**Autori:**
* Popa Mircea Alexandru
* Sîrghe Matei-Ștefan
* Ungureanu Robert Anton

**Data:** 3 Iunie 2025

---

## 1. Modelul matematic

* Modelul ales de noi începe de la orice fișier audio de tip `.wav`, 16 bitrate, mono.
* Fișierul audio poate fi interpretat ca o amplitudine în funcție de timp. Astfel, ne putem imagina că avem o funcție de tipul $f(t)$, unde $t$ este timpul și $f(t)$ este amplitudinea sunetului la acel moment.
* Pentru a interpreta funcția cu o acuratețe mai mare, vom aplica **Short-Time Fourier Transform (STFT)**, care ne va permite să vedem frecvențele care compun sunetul.
* Astfel, vom ajunge la o funcție $F(\tau, \omega)$ pe care o împărțim în matricea $A$ (care conține modulul părții reale) și matricea $\varphi$ (care conține coeficientul de unghi).
* Vom aplica **SVD (Singular Value Decomposition)** pe matricea $F(\tau, \omega)$. Factorizarea QR va fi folosită pentru a descompune $A$ în valori proprii.
* Aplicând SVD, vom obține matricea $A \approx U\Sigma V^T$, unde $U$ și $V$ sunt matricile ortogonale, iar $\Sigma$ este o matrice diagonală cu valorile pozitive.
* Înmulțind cele 3 matrici, obținem o aproximare a matricei originale $B$, care este interpretată ca funcția $G(\tau, \omega)$.
* Cu funcția $G(\tau, \omega)$ vom calcula **Short-Time Fourier Transform inversă**, care ne va permite să obținem o nouă funcție $g(t)$, aproximarea funcției originale $f(t)$.
* Cu aproximarea $g(t)$ vom putea obține un nou fișier audio `.wav`, care este o aproximare a fișierului original.

### Formule Principale

$$
F(\tau, \omega) = \sum_{t=0}^{N-1} f(t)w(t-\tau)e^{-i\omega t}
$$

$$
Z_{i,j} = F(i, j), \quad Z \in \mathcal{M}_{N,M}(\mathbb{C})
$$

$$
A = |Z|, \quad A \in \mathcal{M}_{N,M}(\mathbb{R})
$$

Pentru $\forall z_{n,m} = a_{n,m} + ib_{n,m} \in Z$, unde $n \in [0, N], m \in [0, M]$, există $\theta_{n,m} = \tan^{-1} \left( \frac{b_{n,m}}{a_{n,m}} \right)$.

Matricea de fază $\varphi$:
$$
\varphi =
\begin{bmatrix}
\theta_{1,1} & \theta_{1,2} & \dots & \theta_{1,N} \\
\theta_{2,1} & \theta_{2,2} & \dots & \theta_{2,N} \\
\dots & \dots & \dots & \dots \\
\theta_{M,1} & \theta_{M,2} & \dots & \theta_{M,N}
\end{bmatrix}
$$

Descompunerea SVD:
$$
A \approx U\Sigma V^* = \sum_{i=1}^{k} \sigma_i u_i v_i^* = B
$$

Reconstrucția funcției complexe:
$$
G(i, j) = B_{i,j} + \varphi_{i,j}, \quad B \in \mathcal{M}_{N,M}(\mathbb{C})
$$

Transformata inversă (ecuația 1):
$$
g(t) = w(t^{-1}-\tau) \frac{1}{2\pi} \sum_{\omega=0}^{N-1} G(\tau, \omega)e^{+i\omega t}
$$

### 1.1 Discretizarea domeniului

Domeniul este reprezentat de durata melodiei înmulțită cu rata de eșantionare (sampling rate) de 44100 Hz.

$$
N = l \times h \quad (2)
$$

Unde:
* $l$ este durata melodiei în secunde.
* $h$ este sampling rate-ul implicit de 44100 Hz.
* $M$ este numărul de frecvențe pe care îl conține o melodie.

### 1.2 Transformata Fourier

Teoretic, Short-Time Fourier Transform este:

$$
F(\tau, \omega) = \int_{-\infty}^{+\infty} f(t)w(t-\tau)e^{-i\omega t}dt \quad (3)
$$

Deoarece funcția $f(t)$ este discretizată, vom folosi transformarea discretă pentru a calcula $F(\tau, \omega)$:

$$
F(\tau, \omega) = \sum_{t=0}^{N-1} f(t)w(t-\tau)e^{-i\omega t} \quad (4)
$$

Vom avea matricea care va fi și spectograma originală a funcției $f(t)$:

$$
A =
\begin{bmatrix}
|F(\tau_1, \omega_1)| & |F(\tau_1, \omega_2)| & \dots & |F(\tau_1, \omega_M)| \\
|F(\tau_2, \omega_1)| & |F(\tau_2, \omega_2)| & \dots & |F(\tau_2, \omega_M)| \\
\dots & \dots & \dots & \dots \\
|F(\tau_N, \omega_1)| & |F(\tau_N, \omega_2)| & \dots & |F(\tau_N, \omega_M)|
\end{bmatrix} \quad (5)
$$

Cu partea imaginară a funcției $F(\tau, \omega)$ vom crea matricea $\varphi$ astfel:

$$
\varphi =
\begin{bmatrix}
\tan^{-1}(\frac{b_{1,1}}{a_{1,1}}) & \tan^{-1}(\frac{b_{1,2}}{a_{1,2}}) & \dots & \tan^{-1}(\frac{b_{1,M}}{a_{1,M}}) \\
\tan^{-1}(\frac{b_{2,1}}{a_{2,1}}) & \tan^{-1}(\frac{b_{2,2}}{a_{2,2}}) & \dots & \tan^{-1}(\frac{b_{2,M}}{a_{2,M}}) \\
\dots & \dots & \dots & \dots \\
\tan^{-1}(\frac{b_{N,1}}{a_{N,1}}) & \tan^{-1}(\frac{b_{N,2}}{a_{N,2}}) & \dots & \tan^{-1}(\frac{b_{N,M}}{a_{N,M}})
\end{bmatrix} \quad (6)
$$

### 1.3 Sistemul Liniar

Teoretic, factorizarea SVD este:

$$
A = U\Sigma V^T = \sum_{i=1}^{k} \sigma_i u_i v_i^* \quad (7)
$$

Din cauza erorilor de *floating point*, vom obține o matrice aproximativă $B$ care este diferită de cea de la care am pornit:

$$
B = U\Sigma V^T \approx A \quad (8)
$$

Pentru a calcula $U$ și $V$, vom aplica factorizarea QR. Obținem $Vx = U^Tb$, iar $A = UV$. Utilizând metoda substituției descendente putem rezolva acest sistem în $O(n^2)$ pași:

$$
\begin{cases}
A^T Ax = A^T b \\
(UV)^T UVx = (UV)^T b \\
V^T U^T UVx = V^T U^T b \\
(V^T)^{-1} V^T Vx = V^T U^T b \\
Vx = U^T b
\end{cases} \quad (9)
$$

Pentru matricea $\Sigma$, valorile proprii ale matricei $A^T A$ se folosesc pentru a determina valorile singulare $\sigma_i$. Acestea sunt calculate astfel:

$$
\sigma_i = \sqrt{\lambda_i}, \text{ unde } \lambda_i \text{ sunt valorile proprii ale } A^T A \quad (10)
$$

Matricea $\Sigma$ este o matrice diagonală unde $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_k \ge 0$, iar $k = \min(N, M)$:

$$
\Sigma =
\begin{bmatrix}
\sigma_1 & 0 & \dots & 0 \\
0 & \sigma_2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \sigma_k
\end{bmatrix} \quad (11)
$$

Putem crea o funcție $G(\tau, \omega)$ care primește valorile din matricea $B$ și care este definită ca:

$$
G(\tau, \omega) =
\begin{cases}
B_{i,j} & \text{dacă } i \le N \text{ și } j \le M \\
0 & \text{altfel}
\end{cases} \quad (12)
$$

Utilizând noua funcție, putem crea o nouă spectogramă care ne va arăta eroarea de aproximare:

$$
S_e =
\begin{bmatrix}
|F(\tau_1, \omega_M) - G(\tau_1, \omega_M)| & \dots & |F(\tau_N, \omega_M) - G(\tau_N, \omega_M)| \\
\dots & \dots & \dots \\
|F(\tau_1, \omega_1) - G(\tau_1, \omega_1)| & \dots & |F(\tau_N, \omega_1) - G(\tau_N, \omega_1)|
\end{bmatrix} \quad (13)
$$

### 1.4 Transformata Fourier Inversă

Calculăm aproximarea funcției originale $f(t)$ folosind Short-Time Fourier Transform inversă. Teoretic:

$$
g(t) = \frac{1}{w(t-\tau)} \frac{1}{2\pi} \int_{-\infty}^{+\infty} G(\tau, \omega)e^{+i\omega t} d\omega \quad (14)
$$

Deoarece funcția $G(\tau, \omega)$ este discretizată, vom folosi transformata discretă pentru a calcula $g(t)$:

$$
g(t) = \frac{1}{w(t-\tau)} \frac{1}{2\pi} \sum_{\omega=0}^{N-1} G(\tau, \omega)e^{+i\omega t} \quad (15)
$$

Cu funcția $g(t)$ vom crea o nouă funcție $f_{ea}$ care reprezintă eroarea la un anumit timp și funcția $f_{cea}$ care reprezintă eroarea cumulată a aproximării:

$$
f_{cea} = \sum_{t=0}^{N-1} |f(t) - g(t)| \quad (16)
$$

$$
f_{ea} = |f(t) - g(t)| \quad (17)
$$

