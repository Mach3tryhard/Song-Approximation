import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import time as TIME
import platform
from MetodeNumerice import spline_liniar
from MetodeNumerice import spline_patratic
from MetodeNumerice import spline_cubic

# file format:
# 1 byte - 0=mono, 1=stereo
# 1 byte - size of data type
# 4 bytes - sampling rate
# 4 bytes - data count
# data

def Decompress_Alg(inp,metoda):
    start=TIME.time()
    file_in = inp + '_compressed.bin'
    file_out = inp + '_decompressed.wav'

    with open(file_in, 'rb') as input:
        stereo=int.from_bytes(input.read(1), 'big')
        sizeof_data=int.from_bytes(input.read(1), 'big')
        data_type=np.dtype("int"+str(sizeof_data*8))
        sr=int.from_bytes(input.read(4), 'big')
        data_count=int.from_bytes(input.read(4), 'big')
        data=np.empty(data_count,dtype=data_type)
        if stereo==0:
            for i in range (data_count):
                x=int.from_bytes(input.read(sizeof_data), 'big', signed=True)
                data[i]=x
        if stereo==1:
            left=np.empty(data_count,dtype=data_type)
            for i in range (data_count):
                x=int.from_bytes(input.read(sizeof_data), 'big', signed=True)
                left[i]=x
            right=np.empty(data_count,dtype=data_type)
            for i in range (data_count):
                x=int.from_bytes(input.read(sizeof_data), 'big', signed=True)
                right[i]=x

    target_sr=44100  # target sampling rate for upscaling operation
    scaling=int(target_sr/sr) 

    time = np.arange(data_count) / sr  # original array of points on x axis

    new_data_count=data_count*scaling  # new number of data points
    new_time = np.arange(new_data_count)/target_sr  # new array of points on x axis

    # ------ METODE DE APROXIMARE A FUNCTIEI ------
    print(f"Metoda de interpolare: {metoda}")
    if metoda == 'spline_1':
        if stereo == 0:
            new_data = spline_liniar.spline_liniar(time, data, new_time).astype(data_type)  # new array of points on y axis for mono
        if stereo == 1:
            new_left = spline_liniar.spline_liniar(time, left, new_time).astype(data_type)  # new array of points on y axis for left channel
            new_right=spline_liniar.spline_liniar(time, right, new_time).astype(data_type)  # new array of points on y axis for right channel
    elif metoda == 'spline_2':
        if stereo == 0:
            new_data = spline_patratic.spline_patratic(time, data, new_time).astype(data_type)  # new array of points on y axis for mono
        if stereo == 1:
            new_left = spline_patratic.spline_patratic(time, left, new_time).astype(data_type)  # new array of points on y axis for left channel
            new_right= spline_patratic.spline_patratic(time, right, new_time).astype(data_type)  # new array of points on y axis for right channel
    elif metoda == 'spline_3':
        if stereo == 0:
            new_data = spline_cubic.spline_cubic(time, data, new_time).astype(data_type)  # new array of points on y axis for mono
        if stereo == 1:
            new_left = spline_cubic.spline_cubic(time, left, new_time).astype(data_type)  # new array of points on y axis for left channel
            new_right= spline_cubic.spline_cubic(time, right, new_time).astype(data_type)  # new array of points on y axis for right channel
    else:
        print("Metoda de interpolare necunoscuta!")

    # --------------------------------------------
    if stereo == 1:  # checking for stereo
        reconstructed=np.c_[new_left, new_right]  # adding right channel to reconstruction
    else:
        reconstructed=new_data

    wavfile.write(file_out, target_sr, reconstructed)

    end = TIME.time()
    print(f"Execution time: {end - start:.6f} seconds")

    if platform.system() == 'Windows':
        from multiprocessing import Process
        if stereo == 1:
            p = Process(
                target=plot_show,
                args=(stereo, None, left, right, time, new_time, None, new_left, new_right)
            )
        if stereo == 0:
            p = Process(
                target=plot_show,
                args=(stereo, data, None, None, time, new_time, new_data, None, None)
            )
        p.start()
    elif platform.system() == 'Linux':
        from threading import Thread
        if stereo == 1:
            t = Thread(
                target=plot_show,
                args=(stereo, None, left, right, time, new_time, None, new_left, new_right)
            )
        if stereo == 0:
            t = Thread(
                target=plot_show,
                args=(stereo, data, None, None, time, new_time, new_data, None, None)
            )
        t.start()
        t.join()

def plot_show(stereo, data, left, right, time, new_time, new_data, new_left, new_right):
    if stereo==1:  # stereo plot
        fig, (ax1, ax2)=plt.subplots(2,1,figsize=(10,6))

        ax1.scatter(new_time, new_left, s=20, c='blue', label="New Left Data")
        ax1.scatter(time, left, s=5, c='red', label='Compressed Left Data')

        ax1.plot(new_time, new_left, color='blue', label="Spline Interpolated Left", linewidth=0.8)
        ax1.plot(time, left, color='red', label='Compressed Left', linewidth=0.5)

        ax1.set_xlabel("Time [S]")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Left Channel Decompression")
        ax1.legend(loc='upper right')
        
        ax2.scatter(new_time, new_right, s=20, c='blue', label="New Right Data")
        ax2.scatter(time, right, s=5, c='red', label='Compressed Right Data')

        ax2.plot(new_time, new_right, color='blue', label="Spline Interpolated Right", linewidth=0.8)
        ax2.plot(time, right, color='red', label='Compressed Right', linewidth=0.5)

        ax2.set_xlabel("Time [S]")
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Right Channel Decompression")
        ax2.legend(loc='upper right')
        
        plt.tight_layout()

    else:  # mono plot
        plt.figure(figsize=(10,6))
        plt.scatter(new_time, new_data, s=20, c='blue', label="New Mono Data")
        plt.scatter(time, data, s=5, c='red', label='Compressed Mono Data')
        
        plt.plot(new_time, new_data, color='blue', label="Spline Interpolated Mono", linewidth=0.8)
        plt.plot(time, data, color='red', label='Compressed Mono', linewidth=0.5)

        plt.xlabel("Time [S]")
        plt.ylabel("Amplitude")
        plt.title("Mono Decompression")
        plt.legend(loc='upper right')
        plt.tight_layout()
    plt.show()