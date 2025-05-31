import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import time as TIME
import platform

# file format:
# 1 byte - 0=mono, 1=stereo
# 1 byte - size of data type
# 4 bytes - sampling rate
# 4 bytes - data count
# data

def Compress_Alg(input,force_mono):
    start=TIME.time()

    file_in = input + '.wav'
    file_out = input + '_compressed.bin'

    scaling=10  # scaling factor

    sr, data = wavfile.read(file_in)  # original sampling rate and array of points on y axis

    with open(file_out, 'wb') as output:
        if data.ndim == 1:  # mono audio
            left = None
            right = None
            output.write(bytes([0]))  # 1 byte - 0=mono
            data_type=type(data[0])  # data type for rewriting file
            data_count = len(data)  # original number of data points
        else:  # stereo audio
            if force_mono==True:
                left = data[:, 0]
                right = None
                output.write(bytes([0]))  # 1 byte - 0=mono
                data_type=type(left[0])  # data type for rewriting file
                data_count = len(left)  # original number of data points
            else:
                left = data[:, 0]
                right = data[:, 1]
                output.write(bytes([1]))  # 1 byte - 1=stereo
                data_type=type(left[0])  # data type for rewriting file
                data_count = len(left)  # original number of data points

        sizeof_data=np.dtype(data_type).itemsize
        output.write(bytes([sizeof_data]))  # 1 byte - size of data type
        time = np.arange(data_count) / sr  # original array of points on x axis

    if data.ndim == 1:
        new_data = data[0::scaling]
        new_data_count=len(new_data)  # new number of data points
    elif force_mono == True:
        data_forced = left
        new_data = data_forced[0::scaling]
        new_data_count=len(new_data)  # new number of data points
    else:
        new_left = left[0::scaling]  # new array of points on y axis for left channel / mono
        new_right=right[0::scaling]  # new array of points on y axis for left channel
        new_data_count=len(new_left)  # new number of data points
    new_time = time[0::scaling]  # new array of points on x axis
    new_sr=int(sr/scaling)  # new sampling rate

    with open(file_out, 'ab') as output:
        output.write(new_sr.to_bytes(4, byteorder='big'))  # 4 bytes - sampling rate
        output.write(new_data_count.to_bytes(4, byteorder='big'))  # 4 bytes - data count
        if data.ndim == 1 or force_mono == True:
            for i in range (new_data_count):
                output.write(int(new_data[i]).to_bytes(sizeof_data, byteorder='big', signed=True))  # bytes_size bytes - mono data points
        else:
            for i in range (new_data_count):
                output.write(int(new_left[i]).to_bytes(sizeof_data, byteorder='big', signed=True))  # bytes_size bytes - left channel/mono data points
            for i in range (new_data_count):
                output.write(int(new_right[i]).to_bytes(sizeof_data, byteorder='big', signed=True))  # bytes_size bytes - right channel data points

    end = TIME.time()
    print(f"Execution time: {end - start:.6f} seconds")

    if platform.system() == 'Windows':
        from multiprocessing import Process
        if data.ndim == 1:
            p = Process(
                target=plot_show,
                    args=(data, None, None, time, new_time, new_data, None, None)
            )
        elif force_mono == 1:
            p = Process(
                target=plot_show,
                    args=(data_forced, None, None, time, new_time, new_data, None, None)
            )
        else:
            p = Process(
                target=plot_show,
                    args=(None, left, right, time, new_time, None, new_left, new_right)
            )
        p.start()
    elif platform.system() == 'Linux':
        from threading import Thread
        if data.ndim == 1:
            t = Thread(
                target=plot_show,
                    args=(data, None, None, time, new_time, new_data, None, None)
            )
        elif force_mono == 1:
            t = Thread(
                target=plot_show,
                    args=(data_forced, None, None, time, new_time, new_data, None, None)
            )
        else:
            t = Thread(
                target=plot_show,
                    args=(None, left, right, time, new_time, None, new_left, new_right)
            )
        t.start()
        t.join()

def plot_show(data, left, right, time, new_time, new_data, new_left, new_right):
    if right is not None:  # stereo plot
        fig, (ax1, ax2)=plt.subplots(2,1,figsize=(10,6))

        ax1.scatter(new_time, new_left, s=20, c='blue', label="New Left Data")
        ax1.scatter(time, left, s=5, c='red', label='Original Left Data')
        ax1.plot(time, left, color='red', label='Left', linewidth=0.5)

        ax1.set_xlabel("Time [S]")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Left Channel Compression")
        ax1.legend(loc='upper right')
        
        ax2.scatter(new_time, new_right, s=20, c='blue', label="New Right Data")
        ax2.scatter(time, right, s=5, c='red', label='Original Right Data')
        ax2.plot(time, right, color='red', label='Right', linewidth=0.5)

        ax2.set_xlabel("Time [S]")
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Right Channel Compression")
        ax2.legend(loc='upper right')
        
        plt.tight_layout()

    else:
        plt.figure(figsize=(10,6))
        plt.scatter(new_time, new_data, s=20, c='blue', label="New Mono Data")
        plt.scatter(time, data, s=5, c='red', label='Original Mono Data')
        plt.plot(time, data, color='red', label='Mono', linewidth=0.5)

        plt.xlabel("Time [S]")
        plt.ylabel("Amplitude")
        plt.title("Mono Compression")
        plt.legend(loc='upper right')
        plt.tight_layout()
    plt.show()