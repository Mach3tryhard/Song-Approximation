import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile

# file format:
# 1 byte - 0=mono, 1=stereo
# 1 byte - size of data type
# 4 bytes - sampling rate
# 2 bytes - samples per frame
# 2 bytes - number of frames
# 4 bytes - frequency count:
    # frequency: 
        # index, peak (* number of frames)

frequency_step = 50
frequency_cutoff = 22050
noise_threshold = 1000
input = 'chords_mono'
force_mono = True

def Compression(input, frequency_step, frequency_cutoff, noise_threshold, force_mono):
    file_in = input + '.wav'
    file_out = input + '_compressed.stft'

    sr, data = wavfile.read(file_in)

    with open(file_out, 'wb') as output:

        if data.ndim == 1:  # mono audio
            left = data
            right = None
            output.write(bytes([0]))  # 1 byte - 0=mono
            data_type=type(data[0])  # data type for rewriting file
        else:  # stereo audio
            if force_mono==True:
                left = data[:, 0]
                right = None
                output.write(bytes([0]))  # 1 byte - 0=mono
                data_type=type(left[0])  # data type for rewriting file
            else:
                left = data[:, 0]
                right = data[:, 1]
                output.write(bytes([1]))  # 1 byte - 1=stereo
                data_type=type(left[0])  # data type for rewriting file

        sizeof_data=np.dtype(data_type).itemsize
        output.write(bytes([sizeof_data]))  # 1 byte - size of data type
        output.write(sr.to_bytes(4, byteorder='big'))  # 4 bytes - sampling rate

        samples_per_frame = sr // frequency_step
        output.write(samples_per_frame.to_bytes(2, byteorder='big'))  # 2 bytes - samples per frame

        for k in range (0, 2):
            if k == 0:
                freqs_out, indexes_out, amps_out, data_count = STFT(left, sr, frequency_step, frequency_cutoff, noise_threshold)
                output.write(data_count.to_bytes(2, byteorder='big'))  # 2 bytes - number of frames
                freqs_count = len(freqs_out)
                output.write(freqs_count.to_bytes(4, byteorder='big'))  # 4 bytes - frequency count
                for i in range (freqs_count):
                    output.write(int(freqs_out[i]).to_bytes(sizeof_data, byteorder='big'))  # 2 bytes - frequencies
                    for j in range (data_count):
                        output.write(int(indexes_out[i][j]).to_bytes(sizeof_data, byteorder='big'))  # 2 bytes - peak index
                        output.write(int(amps_out[i][j]).to_bytes(sizeof_data, byteorder='big' ,signed=True))  # 2 bytes - peak amplitude
            elif right is not None:
                freqs_out, indexes_out, amps_out, data_count = STFT(right, sr, frequency_step, frequency_cutoff, noise_threshold)
                freqs_count = len(freqs_out)
                output.write(freqs_count.to_bytes(4, byteorder='big'))  # 4 bytes - frequency count
                for i in range (freqs_count):
                    output.write(int(freqs_out[i]).to_bytes(sizeof_data, byteorder='big'))  # 2 bytes - frequencies
                    for j in range (data_count):
                        output.write(int(indexes_out[i][j]).to_bytes(sizeof_data, byteorder='big'))  # 2 bytes - peak index
                        output.write(int(amps_out[i][j]).to_bytes(sizeof_data, byteorder='big' ,signed=True))  # 2 bytes - peak amplitude


def STFT(data, sr, frequency_step, frequency_cutoff, noise_threshold):


    data_type=type(data[0])
    samples_per_frame = sr // frequency_step
    frequencies, times, Zxx = stft(data, fs=sr, nperseg=samples_per_frame)

    t_window = samples_per_frame / sr
    time_axis_frame = np.linspace(0, t_window, samples_per_frame, endpoint=False)

    frequency_idx_to_keep = np.arange(1, frequency_cutoff // frequency_step + 1)

    freqs_out = []
    indexes_out = []
    amps_out = []

    for freq_idx in frequency_idx_to_keep:
        min_idx = np.zeros(len(times))
        max_idx = np.zeros(len(times))
        min_y = np.zeros(len(times))
        max_y = np.zeros(len(times))
        idx_out = np.zeros(len(times)).astype(data_type)
        val_out = np.zeros(len(times)).astype(data_type)
        
        f = data_type(frequencies[freq_idx])

        if np.max(np.abs(Zxx[freq_idx, :])) > f / noise_threshold:
            for frame_idx, time_start in enumerate(times):
                start_sample = int(time_start * sr) # - samples_per_frame // 2

                if start_sample < 0:
                    continue

                end_sample = start_sample + samples_per_frame

                if end_sample > len(data):
                    continue

                mag = np.abs(Zxx[freq_idx, frame_idx])
                phase = np.angle(Zxx[freq_idx, frame_idx])

                if mag < 1:
                    min_idx[frame_idx] = data_type(samples_per_frame // 4)
                    max_idx[frame_idx] = data_type(samples_per_frame // 4 * 3)
                    min_y[frame_idx]= 0
                    max_y[frame_idx]= 0
                    wave = np.zeros_like(time_axis_frame)
                else:
                    wave = (mag * np.cos(2 * np.pi * f * time_axis_frame + phase))
                    min_idx[frame_idx] = data_type(np.argmin(wave))
                    max_idx[frame_idx] = data_type(np.argmax(wave))
                    min_y[frame_idx] = data_type(round(np.min(wave)))
                    max_y[frame_idx] = data_type(round(np.max(wave)))

                idx_out[frame_idx] = min(min_idx[frame_idx], max_idx[frame_idx])
                val_out[frame_idx] = round(wave[idx_out[frame_idx]])
                # if f>1000:
                #     plt.figure(figsize=(10,6))
                #     plt.plot(time_axis_frame, wave, color='red', label='Mono', linewidth=0.5)
                #     plt.show()
    
            freqs_out.append(f)
            indexes_out.append(idx_out)
            amps_out.append(val_out)
    
    return freqs_out, indexes_out, amps_out, len(times)

Compression(input, frequency_step, frequency_cutoff, noise_threshold, force_mono)