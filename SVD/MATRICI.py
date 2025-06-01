import numpy as np
import librosa
import imageio.v2 as imageio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
from joblib import Parallel, delayed  #Bandaj pt faptul ca nu putem reinveta roata :(
from mpl_toolkits.mplot3d import Axes3D
import os
import STFT

def griffin_lim(magnitude, n_iter=50, window='hann', n_fft=2048, hop_length=None,use_librosa_transforms=False):
    #Griffin_lim : 
    if hop_length is None:
        hop_length = n_fft // 4
        
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    stft = magnitude * angles
    
    for _ in range(n_iter):
        if use_librosa_transforms:
            audio = librosa.istft(stft, hop_length=hop_length, window=window)
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
        else:
            audio = STFT.my_istft(stft)
            stft = STFT.my_stft(audio)
        
        angles = np.exp(1j * np.angle(stft))
        stft = magnitude * angles
        
    if use_librosa_transforms:
        audio = librosa.istft(stft, hop_length=hop_length, window=window)
    else:
        audio = STFT.my_istft(stft)
    return audio

def Gram_Schmidt(A):#   Nu mai este folosit in cod
    #A se va ortogonaliza folosind Gram_Schmidt
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float32)
    R = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        v = A[:, i].astype(np.float32)
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], v)
            v = v - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    
    return Q, R

def Factorizare_QR(A):
    n = A.shape[1]
    Q = A.copy().astype(np.float32)
    R = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] /= R[i, i]
        
        
        def update_column(j):
            if j > i:
                R[i, j] = np.dot(Q[:, i], Q[:, j])
                Q[:, j] -= R[i, j] * Q[:, i]
                
        Parallel(n_jobs=-1)(delayed(update_column)(j) for j in range(n))
    
    return Q, R

def eigen_decomp(A, max_iter=1000, tolerance=1e-8):
    #Descompunere in vectori si valori proprii folosind Factorizare QR, realizata in paralel cu libraria joblib
    B = A.T @ A
    n = B.shape[0]
    V = np.eye(n)
    
    
    def qr_iteration(i):
        nonlocal B, V
        Q, R = Factorizare_QR(B)
        B = R @ Q
        V = V @ Q
    
    Parallel(n_jobs=-1)(delayed(qr_iteration)(i) for i in tqdm(range(100), desc="Eigen Decomp"))
    
    eigenvalues = np.diag(B)
    eigenvectors = V
    return eigenvalues, eigenvectors

def optimised_QR(A):
    n = A.shape[1]
    Q = A.copy().astype(np.float32)
    R = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        norm = np.linalg.norm(Q[:, i])
        if norm < 1e-10: #evitam vectori cu zero
            R[i, i] = 0.0
            continue
        
        R[i, i] = norm
        Q[:, i] /= R[i, i]

        #Optimizare prin vectorizarea operatiilor, should be a lil better
        if i < n - 1:
            R[i, i+1:n] = Q[:, i] @ Q[:, i+1:n] #Produs scalar simultan pt j > i
            Q[:, i+1:n] -= np.outer(Q[:, i], R[i, i+1:n]) #Produsul exterior tot intr-o singura operatie

    return Q, R

def eigen_decomp_optimised(A, max_iter=1000, tolerance=1e-8):
    B = A.T @ A
    n = B.shape[0]
    V = np.eye(n, dtype=np.float32)

    pbar = tqdm(total=max_iter, desc="Eigen Decomp")

    for i in range(max_iter):
        Q, R = optimised_QR(B)
        B = R @ Q
        V = V @ Q

        off_diag = np.abs(B - np.diag(np.diag(B))) #Pt convergenta
        max_off = np.max(off_diag)

        pbar.update(1)
        
        if max_off < tolerance: 
            pbar.set_description(f"Converged la {i+1} iteratii")
            break
    
    pbar.close()
    eigenvalues = np.diag(B)
    eigenvectors = V
    return eigenvalues, eigenvectors


def solve_SVD(spect, k=100):
    #Cod teoretic SVD. Obv nu ruleaza la aceiasi viteza ca np.linalg.svd
    def compute_eigenvectors():
        #lamda, v = eigen_decomp(spect)
        lamda, v = eigen_decomp_optimised(spect)
        return lamda, v
    
    lamda, v = compute_eigenvectors()
    idx = np.argsort(-lamda)
    lamda = lamda[idx]
    v = v[:, idx] #sortam descrescator

    singular_values = np.sqrt(np.maximum(lamda, 0))
    
    # Daca k nu e specificat vom calcula automat pe baza "energiei", adica suma a elementelor matricei cu valori singulare (sigma) la puterea a doua.
    if k == -1:
        energy = np.cumsum(singular_values**2)
        k = np.searchsorted(energy, 0.95 * energy[-1]) + 1 #95% din valorile din matricea sigma. Teoretic acest approach poate scoate noise si alte elemente irelevante din matrice
    
    k = min(k, len(singular_values))
    print(f"{k} Valori proprii")
    
    def compute_u(i):
        if singular_values[i] > 1e-10:
            return (spect @ v[:, i]) / singular_values[i]
        return np.zeros(spect.shape[0])
    
    U = np.column_stack(Parallel(n_jobs=-1)(delayed(compute_u)(i) for i in tqdm(range(k), desc="Computing U")))
    
    return U, singular_values[:k], v[:, :k].T

def svd_wrapper_compression(spectrogram, k,use_numpy_svd=False):
    if use_numpy_svd:
        U, S, Vt = np.linalg.svd(spectrogram, full_matrices=False)
        if k > 0:
            U = U[:, :k]
            S = S[:k]
            Vt = Vt[:k, :]
        return U, S, Vt
    else:
        return solve_SVD(spectrogram, k)

def svd_to_spectrogram(U, S, Vt):
    #Reconstruirea spectogramei A prin imulltirea celor 3 matrici din SVD
    return U @ np.diag(S) @ Vt