from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

sr, data = wavfile.read('3dimineata.wav')
data = data[:10000]  # trim for performance

if data.ndim == 2:
    left = data[:, 0]
    right = data[:, 1]
    time = np.arange(len(left)) / sr

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(time, left, right)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Left Amplitude')
    ax.set_zlabel('Right Amplitude')
    plt.show()