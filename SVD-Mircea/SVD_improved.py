import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallel processing

# Configuration parameters
file_path = 'song_chopped.wav'
nr_valori_singulare = 200
use_griffin_lim = False  # Set to True for phase reconstruction
griffin_lim_iterations = 50  # Number of iterations for Griffin-Lim
use_numpy_svd = False  # Use numpy's faster SVD implementation

def reconstruct_audio(magnitude, phase, sr, output_path="out_SVD_simplu.wav"):
    stft_reconstructed = magnitude * np.exp(1j * phase)
    audio_reconstructed = librosa.istft(stft_reconstructed)
    audio_reconstructed = audio_reconstructed / np.max(np.abs(audio_reconstructed))
    wavfile.write(output_path, sr, (audio_reconstructed * 32767).astype(np.int16))
    return audio_reconstructed

def griffin_lim(magnitude, n_iter=50, window='hann', n_fft=2048, hop_length=None):
    """Reconstruct audio using Griffin-Lim phase estimation"""
    if hop_length is None:
        hop_length = n_fft // 4
        
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    stft = magnitude * angles
    
    for _ in range(n_iter):
        audio = librosa.istft(stft, hop_length=hop_length, window=window)
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
        angles = np.exp(1j * np.angle(stft))
        stft = magnitude * angles
        
    audio = librosa.istft(stft, hop_length=hop_length, window=window)
    return audio

def Gram_Schmidt(A):
    """Orthogonalize matrix A using Gram-Schmidt process"""
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
    """QR factorization using modified Gram-Schmidt"""
    n = A.shape[1]
    Q = A.copy().astype(np.float32)
    R = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] /= R[i, i]
        
        # Process remaining vectors in parallel
        def update_column(j):
            if j > i:
                R[i, j] = np.dot(Q[:, i], Q[:, j])
                Q[:, j] -= R[i, j] * Q[:, i]
                
        Parallel(n_jobs=-1)(delayed(update_column)(j) for j in range(n))
    
    return Q, R

def eigen_decomp(A):
    """Eigen decomposition using QR algorithm with parallel processing"""
    B = A.T @ A
    n = B.shape[0]
    V = np.eye(n)
    
    # Process QR iterations in parallel
    def qr_iteration(i):
        nonlocal B, V
        Q, R = Factorizare_QR(B)
        B = R @ Q
        V = V @ Q
    
    Parallel(n_jobs=-1)(delayed(qr_iteration)(i) for i in tqdm(range(100), desc="Eigen Decomp"))
    
    eigenvalues = np.diag(B)
    eigenvectors = V
    return eigenvalues, eigenvectors

def solve_SVD(spect, k=100):
    """Custom SVD implementation with parallel processing"""
    # Process eigenvalues in parallel
    def compute_eigenvectors():
        lamda, v = eigen_decomp(spect)
        return lamda, v
    
    lamda, v = compute_eigenvectors()
    singular_values = np.sqrt(np.maximum(lamda, 0))
    
    # Determine k if not specified
    if k == -1:
        energy = np.cumsum(singular_values**2)
        k = np.searchsorted(energy, 0.95 * energy[-1]) + 1
    
    k = min(k, len(singular_values))
    
    # Compute U in parallel
    def compute_u(i):
        if singular_values[i] > 1e-10:
            return (spect @ v[:, i]) / singular_values[i]
        return np.zeros(spect.shape[0])
    
    U = np.column_stack(Parallel(n_jobs=-1)(delayed(compute_u)(i) for i in tqdm(range(k), desc="Computing U")))
    
    return U, singular_values[:k], v[:, :k].T

def svd_wrapper_compression(spectrogram, k=100):
    """SVD compression wrapper with choice of implementation"""
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
    """Reconstruct spectrogram from SVD components"""
    return U @ np.diag(S) @ Vt

def process_audio():
    # Read and normalize audio
    sr, data = wavfile.read(file_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        pass
    else:
        raise ValueError(f"Unsupported data type: {data.dtype}")
    
    # Convert to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # Create spectrogram
    stft = librosa.stft(data)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Compress magnitude with SVD
    U, S, Vt = svd_wrapper_compression(magnitude, nr_valori_singulare)
    compressed_magnitude = svd_to_spectrogram(U, S, Vt)
    
    # Handle phase reconstruction
    if use_griffin_lim:
        print("Using Griffin-Lim for phase reconstruction...")
        audio_reconstructed = griffin_lim(compressed_magnitude, griffin_lim_iterations)
    else:
        print("Using original phase...")
        audio_reconstructed = reconstruct_audio(compressed_magnitude, phase, sr)
    
    return audio_reconstructed, sr

# Main processing
if __name__ == "__main__":
    audio_output, sr = process_audio()
    
    # Save final output
    wavfile.write("final_output.wav", sr, (audio_output * 32767).astype(np.int16))
    print("Processing complete. Output saved as 'final_output.wav'")