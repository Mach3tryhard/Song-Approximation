import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa.display #spectograme si waveformuri
from tqdm import tqdm #loading bars cica, might try

#PIPELINE DATA: .WAV -> Preprocessing -> Conversie Matrix ->
#-> Compresie matrix (SVD) -> Reconstruire ->
#-> Postprocessing -> Output Audio


#Variabilele de sub vor trb sa aiba implementare UI
file_path = './samples/singular_chord.wav'
force_mono = True
nr_valori_singulare = 200 #aproape de lossless, 50-100 e ok pt size, sub 50 deja apar artefacte
#poate fi calculata folosind un energy threshold
#nr_valori_singulare = -1 -> calculam noi cat sa fie k, altfel e un parametru manual

def reconstruct_audio(magnitude, phase, sr, output_path="out_SVD_simplu"):
    stft_reconstructed = magnitude * np.exp(1j * phase)

    audio_reconstructed = librosa.istft(stft_reconstructed)

    audio_reconstructed = audio_reconstructed / np.max(np.abs(audio_reconstructed))

    wavfile.write(output_path, sr, audio_reconstructed)

    print(f"written.\n")
    

    return audio_reconstructed

def Gram_Schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float32)
    R = np.zeros((m, n), dtype=np.float32)
    V = A.copy().astype(np.float32)

    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] < 1e-10:
            continue
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i+1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    

    return Q, R



def Factorizare_QR(A):
    n = np.shape(A)[1]
    Q = np.copy(A)
    R = np.zeros((n,n))
    for k in range(0, n, 1):
        for i in range(0, k, 1):
            R[i,k] = Q[:,i] @ Q[:,k]
            Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]
        
        R[k,k] = np.linalg.norm(Q[:,k],2) #norma liniara de ord 2
        
        Q[:,k] = Q[:,k] / R[k,k]
    print('qr ok\n')
    return Q, R


def eigen_decomp(A):
    B = A.T @ A
    #B acum este o matrice patratica, simetrica, pozitiv semi-definita
    n = np.shape(B)[0]
    V = np.eye(n)

    #Factorizare QR/Gram_Schmidt
    print('In descompunerea valorilor proprii\n')
    for i in range(1, 100): #Nu e nevoie de multe iteratii pt descompunere
        Q, R = Gram_Schmidt(B)
        B = R * Q #B conv3rge la o matrice diagonala

        V = V * Q
    
    eigenvalues = np.diag(B)
    eigenvectors = V

    print('valori proprii ok\n')
    return eigenvalues, eigenvectors

def solve_SVD(spect, nr_valori_singulare=100): 
    m, n = np.shape(spect)[0], np.shape(spect)[1]
    lamda, v = eigen_decomp(spect) # lamda = valoare proprie, v = vector propriu
    singular_values = np.sqrt(np.maximum(lamda, 0))
    #valorile singulare sunt definite ca fiind radicalii valorilor proprii

    if nr_valori_singulare == -1: # calculam automat k
        energy = np.cumsum(singular_values**2)
        total_energy = energy[-1]
        k = np.searchsorted(energy, 0.95 * total_energy) + 1 # tinem doar 95% din energy
    else:
        k = min(nr_valori_singulare, len(singular_values))

    U = np.zeros((spect.shape[0], k))
    for i in range(k):
        if singular_values[i] > 1e-10: #sa nu impartim cu 0
            U[:, i] = (spect @ v[:, i]) / singular_values[i]

    print('debug: SVD solved\n')
    return U, singular_values[:k], v[:, :k]



def svd_wrapper_compression(spectrogram):
    print('debug: in wrapper\n')
    U, S, Vt = solve_SVD(spectrogram, nr_valori_singulare)
    return U, S, Vt

def svd_to_spectrogram(U, S, Vt):
    S_matrix = np.diag(S)
    return U @ S_matrix @ Vt.T


def citire_normalizare(file_path, force_mono):
    #Wav-urile in general sunt de tip int16 insa pentru a putea simplicitatea
    #si functionarea corecta a librariei Librosa voi converti la float32
    sr, data = wavfile.read(file_path)
    #print(f"Original shape: {data.shape}, dtype: {data.dtype}")

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0 #genuinely nobody would fucking use this dar at this point i dont even know
    elif data.dtype == np.float32 or data.dtype == np.float64:
        data = data.astype(np.float32)
    else:
        raise ValueError(f"UNsupported data type: {data.dtype}")
    
    if data.ndim == 2 and force_mono:
        data = np.mean(data, axis=1) # convert la mono pe acelasi array, also face media intre cele doua canale



    return sr, data

def create_spectrogram(data, sr):
    #SHORT-TIME-FOURIER-TRANSFORM - din Librosa, maybe fa unul singur later on
    stft = librosa.stft(data)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    return magnitude, phase

def plot_spectogram_mono(magnitude, sr):
    plt.figure(figsize=(10,4))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectogram')
    plt.tight_layout()
    plt.show()

def plot_spectrogram_stereo(left_mag,right_mag, sr):
    #plotam in aceiasi figura
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    img_left = librosa.display.specshow(
        librosa.amplitude_to_db(left_mag, ref=np.max),
        sr=sr, x_axis='time', y_axis='log', ax=axs[0]
    )
    axs[0].set_title("Left Channel")

    img_right = librosa.display.specshow(
        librosa.amplitude_to_db(right_mag, ref=np.max),
        sr=sr, x_axis='time', y_axis='log', ax=axs[1]
    )
    axs[1].set_title("Right Channel")

    fig.colorbar(img_right, ax=axs, format="%+2.0f dB", location="right")
    fig.suptitle('Stereo Spectrograms')

    plt.show()
    

sr, data = citire_normalizare(file_path, force_mono)

if force_mono == False:
    left_data = data[:, 0]
    right_data = data[:, 1]

if data.ndim == 1 or force_mono:
    magnitude, phase = create_spectrogram(data, sr)
    #plot_spectogram_mono(magnitude, sr)
else:
    left_mag, left_phase = create_spectrogram(left_data, sr)
    right_mag, right_phase = create_spectrogram(right_data, sr)
    plot_spectrogram_stereo(left_mag, right_mag, sr)

if force_mono == True:
    U, S, Vt = svd_wrapper_compression(magnitude)
    processed_magnitude_mono = svd_to_spectrogram(U, S, Vt)
    
    audio_reconstructed = reconstruct_audio(processed_magnitude_mono, phase, sr, "out.wav")


else:
    U_st, S_st, Vt_st = svd_wrapper_compression(left_mag)
    U_dr, S_dr, Vt_dr = svd_wrapper_compression(right_mag)
