from mpl_toolkits.mplot3d import Axes3D
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from numpy.polynomial import Polynomial
from scipy.interpolate import make_interp_spline,lagrange

np.set_printoptions(suppress=True)

sr, data = wavfile.read('3dimineata.wav')
scaling=1
k=3
#data = data[:100]  # trim for performance
# data=data/20

if data.ndim ==1: #mono audio
    left = data
    right = None
else: #stereo audio
    left = data[:, 0]
    right = data[:, 1]

data_size = len(left) #spline points
time = np.arange(data_size) / sr # Nr de elemente din array pe sample rate pt a calcula timpul

new_left = left[0::scaling]
if data.ndim !=1:
    new_right=right[0::scaling]
new_time = time[0::scaling]
new_data_size=len(new_left)

if (data_size-1)%scaling != 0:
    new_left=np.append(new_left, left[-1])
    if data.ndim !=1:
        new_right=np.append(new_right, right[-1])
    new_time=np.append(new_time, time[-1])
    new_data_size+=1

spline_left = make_interp_spline(new_time, new_left, k)
reconstructed_left = spline_left(new_time).astype(np.float32)
reconstructed=reconstructed_left
if data.ndim !=1:
    spline_right = make_interp_spline(new_time, new_right, k)
    reconstructed_right = spline_right(new_time).astype(np.float32)
    reconstructed=np.c_[reconstructed_left, reconstructed_right]

print(data)
# print(time)
# print(new_time)
# print(len(time))
# print(len(new_time))
# print(reconstructed)
# print(new_left)

wavfile.write("reconstructed_chords.wav", int(sr/scaling), reconstructed)

# if right is not None: # stereo plot in 3d
#     fig, (ax1, ax2)=plt.subplots(2,1,figsize=(10,6))

#     ax1.scatter(new_time, new_left, s=20, c='blue', label="New Left Data")
#     ax1.scatter(time, left, s=1, c='red', label='Original Left Data')

#     ax1.plot(time, spline_left(time), color='blue', label="Spline Interpolated Left", linewidth=0.8)
#     ax1.plot(time, left, color='red', label='Original Left', linewidth=0.5)

#     ax1.set_xlabel("Time [S]")
#     ax1.set_ylabel("Amplitude")
#     ax1.set_title("Left channel")
#     ax1.legend()
    
#     ax2.scatter(new_time, new_right, s=20, c='blue', label="New Right Data")
#     ax2.scatter(time, right, s=1, c='red', label='Original Right Data')

#     ax2.plot(time, spline_right(time), color='blue', label="Spline Interpolated Right", linewidth=0.8)
#     ax2.plot(time, right, color='red', label='Original Right', linewidth=0.5)

#     ax2.set_xlabel("Time [S]")
#     ax2.set_ylabel("Amplitude")
#     ax2.set_title("Right channel")
#     ax2.legend()
    
#     plt.title("Stereo")
#     plt.tight_layout()
# else:                 # mono plot cu Z = 0
#     plt.figure(figsize=(10,6))
#     plt.scatter(new_time, new_left, s=20, c='blue', label="New Mono Data")
#     plt.scatter(time, left, s=1, c='red', label='Original Mono Data')
    
#     plt.plot(time, spline_left(time), color='blue', label="Spline Interpolated Mono", linewidth=0.8)
#     plt.plot(time, left, color='red', label='Original Mono', linewidth=0.5)

#     plt.xlabel("Time [S]")
#     plt.ylabel("Amplitude")
#     plt.title("Mono")
#     plt.legend()
#     plt.tight_layout()

# plt.show()