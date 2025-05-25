import librosa
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the MP3 (y is a NumPy array, sr is the sample rate)
y, sr = librosa.load("your_file.mp3", sr=None, mono=False)

# 2. If stereo, librosa returns shape=(2, n_samples). 
#    For plotting you can pick one channel or both:
if y.ndim == 2:
    left, right = y
    samples = left  # or combine, or plot both
else:
    samples = y

# 3. Build time axis
time = np.arange(len(samples)) / sr

# 4. Plot
plt.figure(figsize=(12, 4))
plt.plot(time, samples, label="Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.grid(True)
plt.tight_layout()
plt.show()
