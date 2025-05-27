import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# file format:
# 1 byte - 0=mono, 1=stereo
# 1 byte - size of data type
# 4 bytes - sampling rate
# 4 bytes - data count
# data

sr, data = wavfile.read('chords_stereo.wav')  # original sampling rate and array of points on y axis
scaling=5  # scaling factor

#data = data[:10000]  # trim data set

with open('output.bin', 'wb') as output:
    if data.ndim ==1:  # mono audio
        left = data
        right = None
        output.write(bytes([0]))  # 1 byte - 0=mono
    else:  # stereo audio
        left = data[:, 0]
        right = data[:, 1]
        output.write(bytes([1]))  # 1 byte - 1=stereo

    data_type=type(left[0])  # data type for rewriting file
    sizeof_data=np.dtype(data_type).itemsize
    output.write(bytes([sizeof_data]))  # 1 byte - size of data type

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

with open('output.bin', 'ab') as output:
    output.write(new_sr.to_bytes(4, byteorder='big'))  # 4 bytes - sampling rate
    output.write(new_data_count.to_bytes(4, byteorder='big'))  # 4 bytes - data count
    for i in range (new_data_count):
        output.write(int(new_left[i]).to_bytes(sizeof_data, byteorder='big', signed=True))  # bytes_size bytes - left channel/mono data points
    if data.ndim !=1:
        for i in range (new_data_count):
            output.write(int(new_right[i]).to_bytes(sizeof_data, byteorder='big', signed=True))  # bytes_size bytes - right channel data points

print("done")

if right is not None:  # stereo plot
    fig, (ax1, ax2)=plt.subplots(2,1,figsize=(10,6))

    ax1.scatter(new_time, new_left, s=20, c='blue', label="New Left Data")
    ax1.scatter(time, left, s=1, c='red', label='Original Left Data')

    # ax1.plot(new_time, new_left, color='blue', label="Spline Interpolated Left", linewidth=0.8)
    # ax1.plot(time, left, color='red', label='Original Left', linewidth=0.5)

    ax1.set_xlabel("Time [S]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Left channel")
    ax1.legend()
    
    ax2.scatter(new_time, new_right, s=20, c='blue', label="New Right Data")
    ax2.scatter(time, right, s=1, c='red', label='Original Right Data')

    # ax2.plot(new_time, new_right, color='blue', label="Spline Interpolated Right", linewidth=0.8)
    # ax2.plot(time, right, color='red', label='Original Right', linewidth=0.5)

    ax2.set_xlabel("Time [S]")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Right channel")
    ax2.legend()
    
    plt.tight_layout()

else:  # mono plot
    plt.figure(figsize=(10,6))
    plt.scatter(new_time, new_left, s=20, c='blue', label="New Mono Data")
    plt.scatter(time, left, s=1, c='red', label='Original Mono Data')
    
    # plt.plot(new_time, new_left, color='blue', label="Spline Interpolated Mono", linewidth=0.8)
    # plt.plot(time, left, color='red', label='Original Mono', linewidth=0.5)

    plt.xlabel("Time [S]")
    plt.ylabel("Amplitude")
    plt.title("Mono")
    plt.legend()
    plt.tight_layout()

plt.show()