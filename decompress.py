import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import time as TIME

start=TIME.time()

# file format:
# 1 byte - 0=mono, 1=stereo
# 1 byte - size of data type
# 4 bytes - sampling rate
# 4 bytes - data count
# data

input = 'queen'
file_in = input + '_compressed.bin'
file_out = input + '_decompressed.wav'

def spline_liniar(X, Y, x):
    y_val = np.zeros_like(x)
    for i in range(len(X) - 1):
        x0, x1 = X[i], X[i+1]  # coeficienti
        y0, y1 = Y[i], Y[i+1]

        loc_interval = (x >= x0) & (x <= x1)
        #(x >= x0) returneaza un array cu elemente True unde x[] >= x0, la fel si celalalt dar <=. Operatorul & lasa doar o valoare
        
        y_val[loc_interval] = y0 + (y1 - y0) / (x1 - x0) * (x[loc_interval] - x0)

    return y_val

with open(file_in, 'rb') as input:
    stereo=int.from_bytes(input.read(1), 'big')
    sizeof_data=int.from_bytes(input.read(1), 'big')
    data_type=np.dtype("int"+str(sizeof_data*8))
    sr=int.from_bytes(input.read(4), 'big')
    data_count=int.from_bytes(input.read(4), 'big')
    left=np.empty(data_count,dtype=data_type)
    for i in range (0, data_count):
        x=int.from_bytes(input.read(sizeof_data), 'big', signed=True)
        left[i]=x
    if stereo==1:
        right=np.empty(data_count,dtype=data_type)
        for i in range (data_count):
            x=int.from_bytes(input.read(sizeof_data), 'big', signed=True)
            right[i]=x

k=1  # spline order
target_sr=44100  # target sampling rate for upscaling operation
scaling=int(target_sr/sr) 

time = np.arange(data_count) / sr  # original array of points on x axis

new_data_count=data_count*scaling  # new number of data points
new_time = np.arange(new_data_count)/target_sr  # new array of points on x axis

new_left = spline_liniar(time, left, new_time).astype(data_type)  # new array of points on y axis for left channel / mono
if stereo == 1:
    new_right=spline_liniar(time, right, new_time).astype(data_type)  # new array of points on y axis for left channel

if stereo ==1:  # checking for stereo
    reconstructed=np.c_[new_left, new_right]  # adding right channel to reconstruction
else:
    reconstructed=new_left

wavfile.write(file_out, target_sr, reconstructed)

end = TIME.time()
print(f"Execution time: {end - start:.6f} seconds")

if right is not None:  # stereo plot
    fig, (ax1, ax2)=plt.subplots(2,1,figsize=(10,6))

    ax1.scatter(new_time, new_left, s=5, c='blue', label="New Left Data")
    ax1.scatter(time, left, s=20, c='red', label='Original Left Data')

    ax1.plot(new_time, new_left, color='blue', label="Spline Interpolated Left", linewidth=0.8)
    ax1.plot(time, left, color='red', label='Original Left', linewidth=0.5)

    ax1.set_xlabel("Time [S]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Left channel")
    ax1.legend(loc='upper right')
    
    ax2.scatter(new_time, new_right, s=5, c='blue', label="New Right Data")
    ax2.scatter(time, right, s=20, c='red', label='Original Right Data')

    ax2.plot(new_time, new_right, color='blue', label="Spline Interpolated Right", linewidth=0.8)
    ax2.plot(time, right, color='red', label='Original Right', linewidth=0.5)

    ax2.set_xlabel("Time [S]")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Right channel")
    ax2.legend(loc='upper right')
    
    plt.tight_layout()

else:  # mono plot
    plt.figure(figsize=(10,6))
    plt.scatter(new_time, new_left, s=5, c='blue', label="New Mono Data")
    plt.scatter(time, left, s=20, c='red', label='Original Mono Data')
    
    plt.plot(new_time, new_left, color='blue', label="Spline Interpolated Mono", linewidth=0.8)
    plt.plot(time, left, color='red', label='Original Mono', linewidth=0.5)

    plt.xlabel("Time [S]")
    plt.ylabel("Amplitude")
    plt.title("Mono")
    plt.legend(loc='upper right')
    plt.tight_layout()

plt.show()