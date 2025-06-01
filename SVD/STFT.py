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