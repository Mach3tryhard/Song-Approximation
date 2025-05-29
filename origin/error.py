import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

def error_calc(input):
    input = 'chords_mono'
    file_original = input + '.wav'
    file_decompressed = input + '_decompressed.wav'

    sr_original, data_original = wavfile.read(file_original)
    sr_decompressed, data_decompressed = wavfile.read(file_decompressed)

    if data_original.ndim == 1 and data_decompressed.ndim == 1:  # mono audio
        left_original = data_original
        right_original = None
        left_decompressed = data_decompressed
        right_decompressed = None
        if len(left_original) != len(left_decompressed):
            left_decompressed=left_decompressed[:(len(left_original) - len(left_decompressed))]
    elif data_original.ndim == 2 and data_decompressed.ndim == 2:  # stereo audio
        left_original = data_original[:, 0]
        right_original = data_original[:, 1]
        left_decompressed = data_decompressed[:, 0]
        right_decompressed = data_decompressed[:, 1]
        if len(left_original) != len(left_decompressed):
            left_decompressed=left_decompressed[:(len(left_original) - len(left_decompressed))]
            right_decompressed=right_decompressed[:(len(left_original) - len(left_decompressed))]
    else:
        raise RuntimeError("Error: Files do not match")

    data_count_original = len(left_original)
    time_original = np.arange(data_count_original) / sr_original
    data_count_decompressed = len(left_decompressed)
    time_decompressed = np.arange(data_count_decompressed) / sr_decompressed

    if data_original.ndim == 1:
        plt.title('Original vs Decompressed')
        plt.xlabel("Time [S]")
        plt.ylabel("Amplitude")

        plt.grid(True)
        plt.scatter(time_original, left_original, s=5, c='red', label="Original Data")
        plt.scatter(time_decompressed, left_decompressed, s=5, c='blue', label="Decompressed Data")
        plt.plot(time_original, left_original, c='red', linewidth=0.7)
        plt.plot(time_decompressed, left_decompressed, c='blue', linewidth=0.7)
        plt.legend(loc = 'upper right')

        plt.tight_layout()
        plt.show()

        err = np.abs(left_original - left_decompressed)

        plt.title('Error')
        plt.xlabel("Time [S]")
        plt.xlabel("Time [S]")
        plt.plot(time_original, err, 'g', linewidth=0.7)
        plt.legend(loc = 'upper right')

        plt.tight_layout()
        plt.show()
    else:
        plt.title('Original vs Decompressed Left')
        plt.xlabel("Time [S]")
        plt.ylabel("Amplitude")

        plt.grid(True)
        plt.scatter(time_original, left_original, s=5, c='red', label="Original Left Data")
        plt.scatter(time_decompressed, left_decompressed, s=5, c='blue', label="Decompressed Left Data")
        plt.plot(time_original, left_original, c='red', linewidth=0.7)
        plt.plot(time_decompressed, left_decompressed, c='blue', linewidth=0.7)
        plt.legend(loc = 'upper right')

        plt.tight_layout()
        plt.show()

        err_left = np.abs(left_original - left_decompressed)

        plt.title('Error Left')
        plt.xlabel("Time [S]")
        plt.xlabel("Time [S]")
        plt.plot(time_original, err_left, 'g--', linewidth=0.1)
        plt.legend(loc = 'upper right')

        plt.tight_layout()
        plt.show()

        plt.title('Original vs Decompressed Right')
        plt.xlabel("Time [S]")
        plt.ylabel("Amplitude")

        plt.grid(True)
        plt.scatter(time_original, right_original, s=5, c='red', label="Original Right Data")
        plt.scatter(time_decompressed, right_decompressed, s=5, c='blue', label="Decompressed Right Data")
        plt.plot(time_original, right_original, c='red', linewidth=0.7)
        plt.plot(time_decompressed, right_decompressed, c='blue', linewidth=0.7)
        plt.legend(loc = 'upper right')

        plt.tight_layout()
        plt.show()

        err_right = np.abs(right_original - right_decompressed)

        plt.title('Error Right')
        plt.xlabel("Time [S]")
        plt.xlabel("Time [S]")
        plt.plot(time_original, err_right, 'g--', linewidth=0.1)
        plt.legend(loc = 'upper right')

        plt.tight_layout()

        plt.show()