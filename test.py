from mpl_toolkits.mplot3d import Axes3D
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from numpy.polynomial import Polynomial
from scipy.interpolate import make_interp_spline,lagrange
import sys

np.set_printoptions(suppress=True)

def spline_liniar(X, Y):
    y_val = np.zeros_like(X)
    # Pasul 1: Determinam coeficientii polinoamelor liniare pe fiecare subinterval
    for i in range(len(X) - 1):
        x0, x1 = X[i], X[i+1] #coeficienti poly
        y0, y1 = Y[i], Y[i+1]

        loc_interval = (X >= x0) & (X <= x1)
        #(x >= x0) returneaza un array cu elemente True unde x[] >= x0, la fel si celalalt dar <=. Operatorul AND lasa doar o valoare
        
        y_val[loc_interval] = y0 + (y1 - y0) / (x1 - x0) * (X[loc_interval] - x0)

    return y_val

sr, data = wavfile.read('queen.wav')  # original sampling rate and array of points on y axis
scaling=10  # scaling factor
k=3  # order of approximation 
operation=1  # downscaling=0 / upscaling=1
target_sr=44100  # target sampling rate for upscaling operation 

#data = data[:10000]  # trim data set for performance

if data.ndim ==1:  # mono audio
    data_type=type(data[0])  # data type for rewriting file
    left = data
    right = None
else:  # stereo audio
    data_type=type(data[0][0])  # data type for rewriting file
    left = data[:, 0]
    right = data[:, 1]

data_count = len(left)  # original number of data points
time = np.arange(data_count) / sr  # original array of points on x axis

new_left = left[0::scaling]  # new array of points on y axis for left channel / mono
if data.ndim !=1:
    new_right=right[0::scaling]  # new array of points on y axis for left channel
new_time = time[0::scaling]  # new array of points on x axis
new_data_count=len(new_left)  # new number of data points
new_sr=int(sr/scaling)  # new sampling rate

if (data_count-1)%scaling != 0:  # adding the last data point only if it was left out
    new_left=np.append(new_left, left[-1])
    if data.ndim !=1:
        new_right=np.append(new_right, right[-1])
    new_time=np.append(new_time, time[-1])
    new_data_count+=1

reconstructed_left = spline_liniar(new_time, new_left).astype(data_type)
reconstructed=reconstructed_left
if data.ndim !=1:  # checking for stereo
    reconstructed_right = spline_liniar(new_time, new_right).astype(data_type)
    reconstructed=np.c_[reconstructed_left, reconstructed_right]  # adding right channel to reconstruction

# # for debugging:
# print(data)
print(reconstructed)
# print(datatype)
# print(len(reconstructed))
# print(len(data))
# print(spline_left)
# print(time)
# print(new_time)
# print(len(time))
# print(len(new_time))
print(new_left)

wavfile.write("reconstructed_chords.wav", new_sr, reconstructed)

if right is not None: # stereo plot
    fig, (ax1, ax2)=plt.subplots(2,1,figsize=(10,6))

    ax1.scatter(new_time, new_left, s=20, c='blue', label="New Left Data")
    ax1.scatter(time, left, s=1, c='red', label='Original Left Data')

    # ax1.plot(time, spline_left(time), color='blue', label="Spline Interpolated Left", linewidth=0.8)
    ax1.plot(time, left, color='red', label='Original Left', linewidth=0.5)

    ax1.set_xlabel("Time [S]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Left channel")
    ax1.legend()
    
    ax2.scatter(new_time, new_right, s=20, c='blue', label="New Right Data")
    ax2.scatter(time, right, s=1, c='red', label='Original Right Data')

    # ax2.plot(time, spline_right(time), color='blue', label="Spline Interpolated Right", linewidth=0.8)
    ax2.plot(time, right, color='red', label='Original Right', linewidth=0.5)

    ax2.set_xlabel("Time [S]")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Right channel")
    ax2.legend()
    
    plt.tight_layout()

else:                 # mono plot cu Z = 0
    plt.figure(figsize=(10,6))
    plt.scatter(new_time, new_left, s=20, c='blue', label="New Mono Data")
    plt.scatter(time, left, s=1, c='red', label='Original Mono Data')
    
    plt.plot(new_time, reconstructed_left, color='blue', label="Spline Interpolated Mono", linewidth=0.8)
    plt.plot(time, left, color='red', label='Original Mono', linewidth=0.5)

    plt.xlabel("Time [S]")
    plt.ylabel("Amplitude")
    plt.title("Mono")
    plt.legend()
    plt.tight_layout()

plt.show()