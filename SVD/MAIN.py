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
import STFT  #Importam functiile de STFT si IFFT din STFT.py
import MATRICI

# Configs
file_path = './samples/1sec_sample.wav' #Reprezinta fisierul procesat
output_path_name = 'reconstructed_sample.wav'
nr_valori_singulare = 1 # -1 pt calcul automat de k
use_griffin_lim = False  # Daca e true, nu vom folosi phase-ul original ci il vom aproxima cu algoritmul Griffin Lim
griffin_lim_iterations = 50  
use_numpy_svd = False #Self-explanatory.
use_librosa_transforms = True #La fel ca la svd, ori rulam functiile noastre ori functiile librosa

def plot_spectrogram(magnitude, sr, title, filename=None):
    plt.figure(figsize=(12, 6))
    db_spec = librosa.amplitude_to_db(np.abs(magnitude), ref=np.max)
    librosa.display.specshow(db_spec, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    if filename:
        filename = f"./imagini/{filename}"
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
        filename = f"./imagini/{filename}"
        plt.savefig(filename, dpi=150)
    else:
        plt.show()
    plt.close()

def plot_diferenta_spectograma(mag_orig, mag_recon, sr, title, filename=None):
    diff_magnitude = np.abs(mag_orig - mag_recon)

    db_diff = librosa.amplitude_to_db(diff_magnitude, ref=np.max)

    plt.figure(figsize=(12,6))
    librosa.display.specshow(db_diff, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()

    if filename:
        filename = f"./imagini/{filename}"
        plt.savefig(filename, dpi=150)
    else:
        plt.show()
    plt.close()



def plot_3d_spectrogram(magnitude, sr, title, filename=None):
    time = np.linspace(0, magnitude.shape[1] / sr, magnitude.shape[1])
    freq = np.linspace(0, sr / 2, magnitude.shape[0])
    time_grid, freq_grid = np.meshgrid(time, freq)

    db_spec = librosa.amplitude_to_db(magnitude, ref=np.max)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(time_grid, freq_grid, db_spec, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_zlabel('Amplitude (dB)')
    ax.set_title(title)

    if filename:
        filename = f"./imagini/{filename}"
        plt.savefig(filename, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_3d_spectrogram_with_rotation(magnitude, sr, title, output_gif, num_frames=36):
    time = np.linspace(0, magnitude.shape[1] / sr, magnitude.shape[1])
    freq = np.linspace(0, sr / 2, magnitude.shape[0])
    time_grid, freq_grid = np.meshgrid(time, freq)

    db_spec = librosa.amplitude_to_db(magnitude, ref=np.max)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(time_grid, freq_grid, db_spec, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_zlabel('Amplitude (dB)')
    ax.set_title(title)

    #Generare fisiere
    filenames = []
    for angle in range(0, 360, int(360 / num_frames)):
        ax.view_init(elev=30, azim=angle)  #<- Rotatia propriu zisa :D 
        filename = f"./imagini/frame_{angle}.png"
        filenames.append(filename)
        plt.savefig(filename, dpi=150)

    plt.close()

    output_gif = f"./imagini/{output_gif}"
    # Gif cu imageio lib
    with imageio.get_writer(output_gif, mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up
    for filename in filenames:
        os.remove(filename)

    print(f"GIF saved as {output_gif}")


def reconstruct_audio(magnitude, phase, sr):
    stft_reconstructed = magnitude * np.exp(1j * phase)
    if use_librosa_transforms:
        audio_reconstructed = librosa.istft(stft_reconstructed)
    else:
        audio_reconstructed = STFT.my_istft(stft_reconstructed)
    
    audio_reconstructed = audio_reconstructed / np.max(np.abs(audio_reconstructed))
    #wavfile.write(output_path, sr, (audio_reconstructed * 32767).astype(np.int16)) <- acm scriem fisierul din __main__
    return audio_reconstructed

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
        stft = STFT.my_stft(data)

    
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    #SVD
    U, S, Vt = MATRICI.svd_wrapper_compression(magnitude, nr_valori_singulare,use_numpy_svd)
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
    

    compressed_magnitude = MATRICI.svd_to_spectrogram(U, S, Vt)
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
        audio_reconstructed = MATRICI.griffin_lim(compressed_magnitude, griffin_lim_iterations,use_librosa_transforms)
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
    plot_diferenta_spectograma(mag_orig, mag_recon, sr, "Eroarea pe spectrograma", "spectogram_diff.png")
    plot_waveform_comparison(original_data, audio_output, sr, "Mono", "waveform_comparison.png")
    calculate_error_metrics(original_data, audio_output, "Mono")
    

    # plot_3d_spectrogram(mag_orig, sr, "Original Audio Spectrogram", "3d_spectrogram_original.png")
    # plot_3d_spectrogram(mag_recon, sr, "Reconstructed Audio Spectrogram", "3d_spectrogram_reconstructed.png")
    
    plot_3d_spectrogram_with_rotation(mag_orig, sr, "Spectograma Originala", "rotatie_originala.gif")
    plot_3d_spectrogram_with_rotation(mag_recon, sr, "Spectograma Reconstruita", "rotatie_reconstructie.gif")

    #Comment/uncomment diferitele functii de plotare pt diferite versiuni. Fara ultimul parametru de nume de fisier va afisa direct in alt window plot-ul.
    #Plotarea 3d cu rotatie va genera un gif care roteste spectograma :)

    print("\nProcesat :D")