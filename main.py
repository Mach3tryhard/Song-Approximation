from mpl_toolkits.mplot3d import Axes3D
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from numpy.polynomial import Polynomial
from scipy.interpolate import make_interp_spline


sr, data = wavfile.read('chords_mono.wav')
#data = data[:10000]  # trim for performance


if data.ndim ==1: #mono audio
    left = data
    right = None
else: #stereo audio
    left = data[:, 0]
    right = data[:, 1]

data_size = len(left) #spline points

left = left[:data_size]
if right is not None:
    right = right[:data_size]

time = np.arange(len(left)) / sr # Nr de elemente din array pe sample rate pt a calcula timpul

left_norm = left / np.max(np.abs(left)) # Normalizam la [-1, 1] pt interpolare

spline = make_interp_spline(time, left_norm, k = 3)

#dense_time = np.linspace(time[0], time[-1], len(time) * 5)
dense_time = time
reconstructed_norm = spline(dense_time)

reconstructed = np.clip(reconstructed_norm * 32767, -32768, 32767).astype(np.int16)

#wavfile.write("reconstructed_chords.wav", int(sr * 5), reconstructed)
wavfile.write("reconstructed_chords.wav", sr, reconstructed)

#Plotting date initiale
if right is not None: # stereo plot in 3d
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(time, left, right, label="Stereo 3D Waveform")
    ax.scatter(time, left, right, s=1, c='red', label='Stereo 3D Waveform')
    ax.set_xlabel("TIme [s]")
    ax.set_ylabel("Left Channel")
    ax.set_zlabel("Right Channel")
    ax.set_title("3d Stereo")
    ax.legend()
else:                 # mono plot cu Z = 0
    plt.figure(figsize=(10,6))
    #ax.plot(time, left, np.zeros_like(left), label = "Mono Waveform")
    plt.scatter(time, reconstructed, s=1, c='blue', label="Spline", alpha=0.05)
    plt.scatter(time, left, s=1, c='red', label='Mono Waveform')

    plt.xlabel("Time [S]")
    plt.ylabel("Amplitude")
    plt.title("2d Mono")
    plt.legend()
    plt.tight_layout()





plt.show()

