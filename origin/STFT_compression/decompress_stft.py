import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import os
import struct
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from MetodeNumerice import spline_liniar
from MetodeNumerice import spline_patratic
from MetodeNumerice import spline_cubic
import concurrent.futures
from scipy.interpolate import make_interp_spline

# file format:
# 1 byte - 0=mono, 1=stereo
# 4 bytes - initial number of data points
# 4 bytes - maximum value for rescaling 
# 1 byte - size of data type
# 4 bytes - sampling rate
# 2 bytes - samples per frame
# 2 bytes - number of frames
# 4 bytes - frequency count:
    # frequency: 
        # index, peak (* number of frames)

def unpack_and_reconstruct(p):
    return Reconstruct_single_frequency(*p)

def Decompress(inp, metoda, k, plot_freq, plot_singular, plot_entire):
    file_in = inp + '.stft'
    file_out = inp[:-11] + '_decompressed_stft.wav'

    with open(file_in, 'rb') as input:
        stereo=int.from_bytes(input.read(1), 'big')  # 1 byte - 0=mono, 1=stereo
        data_count=int.from_bytes(input.read(4), 'big')  # 4 bytes - number of data points
        data_max=int.from_bytes(input.read(4), 'big')  # 4 bytes - maximum value for rescaling
        sizeof_data=int.from_bytes(input.read(1), 'big')  # 1 byte - size of data type
        data_type=np.dtype("int"+str(sizeof_data*8))
        sr=int.from_bytes(input.read(4), 'big')  # 4 bytes - sampling rate
        samples_per_frame = int.from_bytes(input.read(2), 'big')  # 2 bytes - samples per frame
        frame_count = int.from_bytes(input.read(4), 'big')  # 4 bytes - number of frames
        time = np.arange(data_count) / sr
        if stereo==1:
            freqs_count=int.from_bytes(input.read(4), 'big')  # 4 bytes - frequency count
            freqs=np.empty(freqs_count,dtype=data_type)
            indexes = []
            amps = []
            for i in range (freqs_count):
                x=int.from_bytes(input.read(sizeof_data), 'big')
                freqs[i]=x
                idx = np.empty(frame_count,dtype=data_type)
                val = np.empty(frame_count,dtype=data_type)
                for j in range (frame_count):
                    idx[j] = int.from_bytes(input.read(sizeof_data), 'big')
                    val[j] = struct.unpack('f', input.read(4))[0]
                indexes.append(idx)
                amps.append(val)

            left = Rebuild_waves(time, freqs, indexes, amps, samples_per_frame, frame_count, sr, data_type, metoda, plot_freq, k, plot_singular, plot_entire)
            left = left / np.max(left)
            left = (left * data_max).astype(data_type)
            freqs_count=int.from_bytes(input.read(4), 'big')  # 4 bytes - frequency count
            freqs=np.empty(freqs_count,dtype=data_type)
            indexes = []
            amps = []
            for i in range (freqs_count):
                x=int.from_bytes(input.read(sizeof_data), 'big')
                freqs[i]=x
                idx = np.empty(frame_count,dtype=data_type)
                val = np.empty(frame_count,dtype=data_type)
                for j in range (frame_count):
                    idx[j] = int.from_bytes(input.read(sizeof_data), 'big')
                    val[j] = struct.unpack('f', input.read(4))[0]
                indexes.append(idx)
                amps.append(val)

            right = Rebuild_waves(time, freqs, indexes, amps, samples_per_frame, frame_count, sr, data_type, metoda, plot_freq, k, plot_singular, plot_entire)
            right = right / np.max(right)
            right = (right * data_max).astype(data_type)
            Rewrite(file_out, left, right, sr)

        else:
            freqs_count=int.from_bytes(input.read(4), 'big')  # 4 bytes - frequency count
            freqs=np.empty(freqs_count,dtype=data_type)
            indexes = []
            amps = []
            for i in range (freqs_count):
                x=int.from_bytes(input.read(sizeof_data), 'big')
                freqs[i]=x
                idx = np.empty(frame_count,dtype=data_type)
                val = np.empty(frame_count,dtype=data_type)
                for j in range (frame_count):
                    idx[j] = int.from_bytes(input.read(sizeof_data), 'big')
                    val[j] = struct.unpack('f', input.read(4))[0]
                indexes.append(idx)
                amps.append(val)

            left = Rebuild_waves(time, freqs, indexes, amps, samples_per_frame, frame_count, sr, data_type, metoda, plot_freq, k, plot_singular, plot_entire)
            left = left / np.max(left)
            left = (left * data_max).astype(data_type)
            Rewrite(file_out, left, None, sr)

def Reconstruct_single_frequency(time, f, indexes, amps, samples_per_frame, frame_count, sr, time_axis_frame, data_count, metoda, window, plot_freq, k, plot_singular, plot_entire):
    wave_component = np.zeros(data_count)
    for i in range (frame_count):
        start_sample = samples_per_frame * (i) // 2 - samples_per_frame // 2
        
        end_sample = start_sample + samples_per_frame

        if start_sample > data_count:
            continue
        
        available = samples_per_frame
        if end_sample > data_count:
            available = data_count - start_sample
            end_sample = data_count
        
        frame_x_idx = []
        frame_x = []
        frame_y = []
        jump=sr//int(4*f)

        origin = int(indexes[i])
        nr_of_jumps_l = 0
        nr_of_jumps_r = 0

        while origin > jump:
            origin -= jump
            nr_of_jumps_l += 1
        origin = indexes[i]

        while origin < samples_per_frame - jump:
            origin += jump
            nr_of_jumps_r += 1
        origin = indexes[i]

        for j in range (nr_of_jumps_l, -1, -1):
            frame_x_idx.append(origin - j * jump)
            if j % 4 == 1 or j % 4 == 3:
                frame_y.append(0)
            elif j % 4 == 2:
                frame_y.append(-amps[i])
            else:
                frame_y.append(amps[i])

        for j in range (1, nr_of_jumps_r + 1):
            frame_x_idx.append(origin + j * jump)
            if j % 4 == 1 or j % 4 == 3:
                frame_y.append(0)
            elif j % 4 == 2:
                frame_y.append(-amps[i])
            else:
                frame_y.append(amps[i])

        for j in range (len(frame_x_idx)):
            frame_x.append(time_axis_frame[frame_x_idx[j]])
        
        frame_x=np.asarray(frame_x)
        frame_y=np.asarray(frame_y)

        if metoda == 'spline1':
            wave = spline_liniar.spline_liniar(frame_x, frame_y, time_axis_frame[:available])
        elif metoda == 'spline2':
            wave = spline_patratic.spline_patratic(frame_x, frame_y, time_axis_frame[:available])
        elif metoda == 'spline3':
            wave = spline_cubic.spline_cubic(frame_x, frame_y, time_axis_frame[:available])
        elif metoda == 'np_spline':
            spline = make_interp_spline(frame_x, frame_y, k=k)
            wave = spline(time_axis_frame[:available])

        for j in range (frame_x_idx[1]):
            if j + (frame_x_idx[1] - j) * 2 < available:
                if frame_y[1] == 0:
                    wave[j] = -wave[j + (frame_x_idx[1] - j) * 2]
                else:
                    wave[j] = wave[j + (frame_x_idx[1] - j) * 2]
        for j in range (frame_x_idx[-2]+1, samples_per_frame+1):
            if j < available:
                if frame_y[-2] == 0:
                    wave[j] = -wave[j - (j - frame_x_idx[-2]) * 2]
                else:
                    wave[j] = wave[j - (j - frame_x_idx[-2]) * 2]

        if np.max(wave) > 50 and i == 25 and f == plot_freq and plot_singular == True:
            plt.figure(figsize=(10,6))
            plt.plot(time_axis_frame, wave, label = "Inainte de window")
            plt.scatter(frame_x, frame_y, s=50, label = "Puncte interpoalre")
            plt.title(f"Frecventa: {f} Hz la frame-ul {i}")
            plt.xlabel("Timp (s)")
            plt.ylabel("Amplitudine")
            plt.legend(loc = 'upper right')
            plt.show()

        if start_sample < 0:
            wave = np.zeros(samples_per_frame + start_sample)
            wave_component[:end_sample]+=wave
        else:
            wave[:available] *= window[:available]
            wave_component[start_sample:end_sample]+=wave[:available]

        if np.max(wave) > 50 and i == 25 and f == plot_freq and plot_singular == True:
            plt.figure(figsize=(10,6))
            plt.plot(time_axis_frame, wave, label = "Dupa window")
            plt.scatter(frame_x, frame_y, s=50, label = "Puncte interpoalre")
            plt.title(f"Frecventa: {f} Hz la frame-ul {i}")
            plt.xlabel("Timp (s)")
            plt.ylabel("Amplitudine")
            plt.legend(loc = 'upper right')
            plt.show()
        
    if f == plot_freq and plot_entire == True:
        plt.figure(figsize=(10,6))
        plt.plot(time[:end_sample], wave_component[:end_sample], label = "Waveform")
        plt.title(f"Grafic pentru Frecventa: {f} Hz")
        plt.xlabel("Timp (s)")
        plt.ylabel("Amplitudine")
        plt.legend(loc = 'upper right')
        plt.show()

    return wave_component

def Rebuild_waves(time, freqs, indexes, amps, samples_per_frame, frame_count, sr, data_type, metoda, plot_freq, k, plot_singular, plot_entire):
    t_window = samples_per_frame / sr
    time_axis_frame = np.linspace(0, t_window, samples_per_frame, endpoint=False)
    data_count = len(time)

    window = np.hanning(samples_per_frame) ** (35/100)

    rebuilt_data = np.zeros(data_count)

    args = [
        (time, f, indexes[i], amps[i], samples_per_frame, frame_count, sr, time_axis_frame, data_count, metoda, window, plot_freq, k, plot_singular, plot_entire)
        for i, f in enumerate(freqs)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(unpack_and_reconstruct, args)
        for i, wave_component in enumerate(results):
            print(freqs[i] / np.max(freqs) * 100, "%", "done")
            rebuilt_data += wave_component

    return rebuilt_data.astype(data_type)

def Rewrite(file_out, left, right, sr):
    if right is None:
        wavfile.write(file_out, sr, left)
    else:
        stereo_data = np.stack((left, right), axis=-1)
        wavfile.write(file_out, sr, stereo_data)