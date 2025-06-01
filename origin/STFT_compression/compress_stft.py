import numpy as np
from scipy.signal import stft
from scipy.io import wavfile
import struct
import concurrent.futures

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

    return single_frequency(*p)

def Compression(input, force_mono, frequency_step, frequency_cutoff, noise_threshold):

    file_in = input + '.wav'
    file_out = input + '_compressed.stft'
    file_out_direct = input + '_direct.wav'

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

        data_count = len(left)

        output.write(data_count.to_bytes(4, byteorder='big'))  # 4 bytes - initial number of data points
        output.write(int(np.max(data)).to_bytes(4, byteorder='big'))  # 4 bytes - maximum value for rescaling 

        sizeof_data=np.dtype(data_type).itemsize

        output.write(bytes([sizeof_data]))  # 1 byte - size of data type
        output.write(sr.to_bytes(4, byteorder='big'))  # 4 bytes - sampling rate

        samples_per_frame = sr // frequency_step
        output.write(samples_per_frame.to_bytes(2, byteorder='big'))  # 2 bytes - samples per frame

        for k in range (0, 2):
            if k == 0:
                freqs_out, indexes_out, amps_out, frame_count, new_left = STFT(file_out_direct, left, sr, frequency_step, frequency_cutoff, noise_threshold)
                output.write(frame_count.to_bytes(4, byteorder='big'))  # 4 bytes - number of frames
                freqs_count = len(freqs_out)
                output.write(freqs_count.to_bytes(4, byteorder='big'))  # 4 bytes - frequency count

                for i in range (freqs_count):
                    output.write(int(freqs_out[i]).to_bytes(sizeof_data, byteorder='big'))  # 2 bytes - frequencies
                    
                    for j in range (frame_count):
                        output.write(int(indexes_out[i][j]).to_bytes(sizeof_data, byteorder='big'))  # 2 bytes - peak index
                        amps_bytes = struct.pack('f', amps_out[i][j])
                        output.write(amps_bytes)  # 4 bytes - peak amplitude

            elif right is not None and force_mono is not True:
                freqs_out, indexes_out, amps_out, frame_count, new_right = STFT(file_out_direct, right, sr, frequency_step, frequency_cutoff, noise_threshold)
                freqs_count = len(freqs_out)
                output.write(freqs_count.to_bytes(4, byteorder='big'))  # 4 bytes - frequency count

                for i in range (freqs_count):
                    output.write(int(freqs_out[i]).to_bytes(sizeof_data, byteorder='big'))  # 2 bytes - frequencies

                    for j in range (frame_count):
                        output.write(int(indexes_out[i][j]).to_bytes(sizeof_data, byteorder='big'))  # 2 bytes - peak index
                        amps_bytes = struct.pack('f', amps_out[i][j])
                        output.write(amps_bytes)  # 4 bytes - peak amplitude

        Rewrite(file_out_direct, new_left, new_right, sr)

def Rewrite(file_out, left, right, sr):

    if right is None:
        wavfile.write(file_out, sr, left)

    else:
        stereo_data = np.stack((left, right), axis=-1)
        wavfile.write(file_out, sr, stereo_data)

def single_frequency(times, sr, samples_per_frame, data_count, data_type, frequencies, time_axis_frame, Zxx, freq_idx, window, noise_threshold):

    min_idx = np.zeros(len(times))

    max_idx = np.zeros(len(times))
    min_y = np.zeros(len(times))
    max_y = np.zeros(len(times))

    idx_out = np.zeros(len(times)).astype(data_type)
    val_out = np.zeros(len(times)).astype(float)
    
    f = data_type(frequencies[freq_idx])

    component = np.zeros(data_count)

    if np.max(np.abs(Zxx[freq_idx, :])) > f / noise_threshold:

        for frame_idx, time_start in enumerate(times):

            start_sample = int(time_start * sr) - samples_per_frame // 2

            if start_sample < 0:
                continue

            end_sample = start_sample + samples_per_frame
            available = samples_per_frame

            if end_sample > data_count:
                available = data_count - start_sample
                end_sample = data_count

            mag = np.abs(Zxx[freq_idx, frame_idx])
            phase = np.angle(Zxx[freq_idx, frame_idx])

            if mag < 0.01:
                min_idx[frame_idx] = data_type(samples_per_frame // 4)
                max_idx[frame_idx] = data_type(samples_per_frame // 4 * 3)
                min_y[frame_idx]= 0
                max_y[frame_idx]= 0
                wave = np.zeros_like(time_axis_frame)

            else:
                wave = (mag * np.cos(2 * np.pi * f * time_axis_frame + phase))
                min_idx[frame_idx] = data_type(np.argmin(wave))
                max_idx[frame_idx] = data_type(np.argmax(wave))
                min_y[frame_idx] = np.min(wave)
                max_y[frame_idx] = np.max(wave)

            idx_out[frame_idx] = min(min_idx[frame_idx], max_idx[frame_idx])
            val_out[frame_idx] = wave[idx_out[frame_idx]]

            wave[:available] *= window[:available]
            component[start_sample:end_sample]+=wave[:available]
        
        return f, idx_out, val_out, component


def STFT(file_out_direct, data, sr, frequency_step, frequency_cutoff, noise_threshold):

    data_type=type(data[0])
    samples_per_frame = sr // frequency_step

    window = np.sin(np.pi * np.linspace(0, 1, samples_per_frame))

    frequencies, times, Zxx = stft(data, fs=sr, nperseg=samples_per_frame ,window=window)

    t_window = samples_per_frame / sr
    time_axis_frame = np.linspace(0, t_window, samples_per_frame, endpoint=False)

    frequency_idx_to_keep = np.arange(1, frequency_cutoff // frequency_step + 1)

    freqs_out = []
    indexes_out = []
    amps_out = []
    reconstructed=np.zeros(len(data))
    data_count = len(data)

    args = [
        (times, sr, samples_per_frame, data_count, data_type, frequencies, time_axis_frame, Zxx, freq_idx, window, noise_threshold)
        for freq_idx in frequency_idx_to_keep
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        results = executor.map(unpack_and_reconstruct, args)
        for i, result in enumerate(results):
            if result == None:
                continue
            f, idx_out, val_out, component = result
            print(frequencies[frequency_idx_to_keep[i]] / np.max(frequencies) * 100, "%", "done")
            freqs_out.append(f)
            indexes_out.append(idx_out)
            amps_out.append(val_out) 
            reconstructed+=component

    reconstructed = (reconstructed).astype(data_type)
    return freqs_out, indexes_out, amps_out, len(times), reconstructed