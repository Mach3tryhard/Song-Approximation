import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
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

def Decompress(inp, metoda):
    file_in = inp + '_compressed.stft'
    file_out = inp + '_decompressed_stft.wav'

    with open(file_in, 'rb') as input:
        stereo=int.from_bytes(input.read(1), 'big')  # 1 byte - 0=mono, 1=stereo
        sizeof_data=int.from_bytes(input.read(1), 'big')  # 1 byte - size of data type
        data_type=np.dtype("int"+str(sizeof_data*8))
        sr=int.from_bytes(input.read(4), 'big')  # 4 bytes - sampling rate
        samples_per_frame = int.from_bytes(input.read(2), 'big')  # 2 bytes - samples per frame
        frame_count = int.from_bytes(input.read(2), 'big')  # 2 bytes - number of frames
        if stereo==0:
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
                    val[j] = int.from_bytes(input.read(sizeof_data), 'big', signed=True)
                indexes.append(idx)
                amps.append(val)
            Rebuild_waves(freqs, indexes, amps, samples_per_frame, frame_count, sr)

def Rebuild_waves(freqs, indexes, amps, samples_per_frame, frame_count, sr):
    t_window = samples_per_frame / sr
    time_axis_frame = np.linspace(0, t_window, samples_per_frame, endpoint=False)

    for f in freqs:
        for i in range (frame_count):
            start_sample = samples_per_frame * (i) // 2

            if start_sample < 0:
                continue
            
            end_sample = start_sample + samples_per_frame

            if end_sample > frame_count * samples_per_frame // 2:
                continue

            frame_x = []
            frame_y = []
            frame_x.append(time_axis_frame[indexes[f][i]])
            frame_y.append(amps[f][i])

            if amps[f][i] < 0:


Decompress("chords_mono", None)