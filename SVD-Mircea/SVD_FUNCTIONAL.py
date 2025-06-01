import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
from joblib import Parallel, delayed  #Bandaj pt faptul ca nu putem reinveta roata :(
import os


# Configs
file_path = '1sec_sample.wav'
output_path_name = 'reconstructed_sample.wav'
nr_valori_singulare = -1 # -1 pt calcul automat de k
use_griffin_lim = False  # Daca e true, nu vom folosi phase-ul original ci il vom aproxima cu algoritmul Griffin Lim
griffin_lim_iterations = 50  
use_numpy_svd = False #Self-explanatory.
use_librosa_transforms = True #La fel ca la svd, ori rulam functiile noastre ori functiile librosa
#write_binary = True #scrie matriciile SVD intr-un fisier binar
#read_binary = True // nu e worth, aparent scrii mai mult decat fisierul original

#Pt un cod rulabil a nu se utiliza simultan si functiile proprii fourier si svd, timpul de procesare este prea mare

def my_fft_recursiv(x):
    #Algoritmul Cooley-Tukey recursiv, Radix-2
    #Inputul trebuie sa fie len(x) = k^2

    N = len(x)
    if N == 1:
        return x
    elif N % 2 != 0:
        raise ValueError("Cum naiba ai intrat cu o valoare cu un length impar")
    
    x_even = my_fft_recursiv(x[::2])
    x_odd = my_fft_recursiv(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([
        x_even + factor[:N//2] * x_odd,
        x_even - factor[:N//2] * x_odd
    ])

def my_stft(signal, fft_size=1024, hop_size=512, window_fn=np.hanning):
    #SHort-Time Fourier Transform (STFT) al unui semnal audio cu windowing de tip Hann
    window = window_fn(fft_size)
    num_frames = 1 + (len(signal) - fft_size) // hop_size
    #// -> floor division, divizie care da round la rezultat
    stft_matrix = np.empty((num_frames, fft_size), dtype=np.complex64)

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start+fft_size] * window
        stft_matrix[i, :] = my_fft_recursiv(frame)
    
    return stft_matrix

def my_ifft_recursiv(x):
    x_conj = np.conj(x)
    x = my_fft_recursiv(x_conj)
    return np.conj(x) / len(x)

def my_istft(stft_matrix, fft_size=1024, hop_size=512, window_fn=np.hanning):
    window = window_fn(fft_size)
    num_frames = stft_matrix.shape[0]
    length_signal = fft_size + hop_size * (num_frames -1)
    signal = np.zeros(length_signal)
    window_sum = np.zeros(length_signal)

    for i in range(num_frames):
        start = i * hop_size
        frame_time = np.real(my_ifft_recursiv(stft_matrix[i, :]))
        signal[start:start+fft_size] += frame_time * window
        window_sum[start:start+fft_size] += window ** 2
    
    window_sum[window_sum < 1e-10] = 1e-10
    signal /= window_sum

    return signal

def plot_spectrogram(magnitude, sr, title, filename=None):
    plt.figure(figsize=(12, 6))
    db_spec = librosa.amplitude_to_db(np.abs(magnitude), ref=np.max)
    librosa.display.specshow(db_spec, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()
    plt.close()

def plot_waveform_comparison(original, reconstructed, sr, channel_name, filename=None):
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    time = np.arange(min_len) / sr

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time, original, 'b-', alpha=0.7, label='Original')
    plt.plot(time, reconstructed, 'r-', alpha=0.5, label='Reconstructed')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform Comparison - {channel_name} Channel')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(3, 1, 2)
    amplitude_diff = np.abs(original - reconstructed)
    plt.plot(time, amplitude_diff, 'g-')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude Difference')
    plt.title('Absolute Error')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(3, 1, 3)
    cumulative_error = np.cumsum(amplitude_diff)
    plt.plot(time, cumulative_error, 'm-')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.title('Cumulative Error')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()
    plt.close()

def calculate_error_metrics(original, reconstructed, channel_name):
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    error = original - reconstructed

    mae = np.mean(np.abs(error)) #pe engleza, mean absolute error
    max_error = np.max(np.abs(error))
    snr = 20 * np.log10(np.linalg.norm(original) / np.linalg.norm(error)) if np.linalg.norm(error) > 0 else float('inf')

    print(f"\nErori - {channel_name} Channel:")
    print(f"  Eroarea absoluta(medie): {mae:.6f}")
    print(f"  Eroarea maxima: {max_error:.6f}")
    print(f"  Signal-to-Noise Ratio: {snr:.2f} dB")

    return {'mae': mae, 'max_error': max_error, 'snr': snr}



def plot_waveform_comparison(original, reconstructed, sr, channel_name, filename=None):
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    time = np.arange(min_len) / sr

    plt.figure(figsize=(14, 12))

    
    plt.subplot(4, 1, 1)
    plt.plot(time, original, 'b-', alpha=0.7, label='Original')
    plt.plot(time, reconstructed, 'r-', alpha=0.5, label='Reconstructed')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Comparatie Waveform - {channel_name} Channel (Full Duration)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Eroarea absoluta
    plt.subplot(4, 1, 2)
    amplitude_diff = np.abs(original - reconstructed)
    plt.plot(time, amplitude_diff, 'g-')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude Difference')
    plt.title('Absolute Error')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Eroarea cumulativa
    plt.subplot(4, 1, 3)
    cumulative_error = np.cumsum(amplitude_diff)
    plt.plot(time, cumulative_error, 'm-')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.title('Cumulative Error')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Zoomed-in
    zoom_duration = 0.2 #secunde
    center_idx = min_len // 2
    zoom_samples = int(zoom_duration * sr)
    start = max(center_idx - zoom_samples // 2, 0)
    end = min(center_idx + zoom_samples // 2, min_len)
    zoom_time = time[start:end]

    plt.subplot(4, 1, 4)
    plt.plot(zoom_time, original[start:end], 'b-', label='Original')
    plt.plot(zoom_time, reconstructed[start:end], 'r-', alpha=0.6, label='Reconstructed')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Zoomed-In Comparison ({zoom_duration:.1f}s window around midpoint)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        plt.show()
    plt.close()



def reconstruct_audio(magnitude, phase, sr):
    stft_reconstructed = magnitude * np.exp(1j * phase)
    if use_librosa_transforms:
        audio_reconstructed = librosa.istft(stft_reconstructed)
    else:
        audio_reconstructed = my_istft(stft_reconstructed)
    
    audio_reconstructed = audio_reconstructed / np.max(np.abs(audio_reconstructed))
    #wavfile.write(output_path, sr, (audio_reconstructed * 32767).astype(np.int16)) <- acm scriem fisierul din __main__
    return audio_reconstructed

def griffin_lim(magnitude, n_iter=50, window='hann', n_fft=2048, hop_length=None):
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
            audio = my_istft(stft)
            stft = my_stft(audio)
        
        angles = np.exp(1j * np.angle(stft))
        stft = magnitude * angles
        
    if use_librosa_transforms:
        audio = librosa.istft(stft, hop_length=hop_length, window=window)
    else:
        audio = my_istft(stft)
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
    #Cod neoptimizat de SVD. Foarte incet
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

def svd_wrapper_compression(spectrogram, k):
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

def process_audio():
    #Pe langa citire se normalizeaza data la [-1, 1]
    sr, data = wavfile.read(file_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        pass
    else:
        raise ValueError(f"Unsupported data type: {data.dtype}")
    
    # Daca e stereo convertim la mono printr-o medie a celor doua canale
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    if use_librosa_transforms:
        stft = librosa.stft(data)
    else:
        stft = my_stft(data)

    
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    #SVD
    U, S, Vt = svd_wrapper_compression(magnitude, nr_valori_singulare)
    #print(f"{type(U), len(U), type(S), len(S), type(Vt), len(Vt)}") #debug
    print(f"{type(U[0][0]), type(S[0]), type(Vt[0][0])}")
    print(f"{np.shape(U), np.shape(S), np.shape(Vt)}")

    #File format
    #U e o matrice m x 
    #Primul si al doilea int sunt marimea matricii U
    #Al treilea int e dimensiunea S
    #Al patrulea int e dimensiunea Vt
    #Restul valori sunt in format float32, datele din toate cele 3 matrici

    #if write_binary == True:   
    #   with open('./binary/SVD_arrays.bin', 'wb') as f:
    #        #lengthurile
    #        f.write(np.array(U.shape, dtype=np.int32).tobytes())
    #        f.write(np.array([len(S)], dtype=np.int32).tobytes())
    #        f.write(np.array(Vt.shape, dtype=np.int32).tobytes())

    #        f.write(U.astype(np.float32).tobytes())
    #        f.write(S.astype(np.float32).tobytes())
    #        f.write(Vt.astype(np.float32).tobytes())
    #Deprecated, aparent sa scrii matriciile ale svd-ului ocupa mai multa memorie decat melodia/reconstructia ei.
    

    compressed_magnitude = svd_to_spectrogram(U, S, Vt)
    #sanity check ca mor
    if compressed_magnitude.shape != magnitude.shape:
        print(f"Original {magnitude.shape}, Compressed {compressed_magnitude.shape}")
        min_rows = min(compressed_magnitude.shape[0], magnitude.shape[0])
        min_cols = min(compressed_magnitude.shape[1], magnitude.shape[1])
        compressed_magnitude = compressed_magnitude[:min_rows, :min_cols]
        phase = phase[:min_rows, :min_cols]
    
    
    
    #phase reconstruction sau nah
    if use_griffin_lim:
        print("Griffin Lim pentru reconstruire...")
        audio_reconstructed = griffin_lim(compressed_magnitude, griffin_lim_iterations)
    else:
        print("Folosim faza originala...")
        audio_reconstructed = reconstruct_audio(compressed_magnitude, phase, sr)
    
    return audio_reconstructed, sr


if __name__ == "__main__":
    audio_output, sr = process_audio()

    
    wavfile.write(output_path_name, sr, (audio_output * 32767).astype(np.int16))
    print(f"Processing complete. Output saved as {output_path_name}")


    sr_orig, original_data = wavfile.read(file_path)
    if original_data.dtype == np.int16:
        original_data = original_data.astype(np.float32) / 32768.0
    elif original_data.dtype == np.int32:
        original_data = original_data.astype(np.float32) / 2147483648.0
    elif original_data.dtype == np.float32:
        pass
    else:
        raise ValueError(f"Unsupported data type: {original_data.dtype}")

    if original_data.ndim > 1:
        original_data = np.mean(original_data, axis=1)

    #aici nu voi folosi functiile proprii, e doar pt plotting
    stft_orig = librosa.stft(original_data)
    stft_recon = librosa.stft(audio_output)
    mag_orig = np.abs(stft_orig)
    mag_recon = np.abs(stft_recon)

    print("\nGenerare imagini...")

    plot_spectrogram(mag_orig, sr, "Original Spectrogram", "spectrogram_original.png")
    plot_spectrogram(mag_recon, sr, "Reconstructed Spectrogram", "spectrogram_reconstructed.png")
    plot_waveform_comparison(original_data, audio_output, sr, "Mono", "waveform_comparison.png")
    calculate_error_metrics(original_data, audio_output, "Mono")

    print("\nProcesat :D")