## This file consists of extracting features from the data


import numpy as np
import copy
import math
import ntpath
from scipy.fftpack import fft
from scipy.io import wavfile
from numpy.lib import stride_tricks
from python_speech_features.sigproc import framesig
import os
#from pydub import AudioSegment

"""
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features.sigproc import framesig
from python_speech_features.sigproc import powspec
"""
from matplotlib.pyplot import specgram

import matplotlib.pyplot as plt
import wave
import contextlib

def stereoToMono(audiodata):
    newaudiodata = []

    for i in range(len(audiodata)):
        d = audiodata[i][0]/2 + audiodata[i][1]/2
        newaudiodata.append(d)

    return np.array(newaudiodata, dtype='int16')

def stereoToMonoUtil():
    """
    audiofiles = [os.path.join(root, name)
             for root, dirs, files in os.walk(filepath)
             for name in files
             if name.endswith((".wav"))]

    for audio in audiofiles:
        head, tail = ntpath.split(filename)
        newaudiodata = stereoToMono(audio)
        newfilename = tail
        #wavfile.write(newfilename, sr, newaudiodata)
    """
    audio_path = "/home/aiswarya/SNN_works/my_code/spoken-digit-dataset/free-spoken-digit-dataset-master/recordings"

    # Gets list of all audio files in the directory
    audio = [os.path.join(root, name)
             for root, dirs, files in os.walk(audio_path)
             for name in files
             if name.endswith((".wav"))]

    for path in audio:
        sound = AudioSegment.from_wav(path)
        sound = sound.set_channels(1)
        sound.export(path, format="wav")

#stereoToMonoUtil()
"""
def mel_Freq(file_name):

    (rate, sig) = wavfile.read(file_name)

    with contextlib.closing(wave.open(file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    number_of_frames = 40
    samples_per_frame = len(sig) / number_of_frames

    win_length = duration / number_of_frames

    i = 0
    x = 1
    frames = []
    frame_sample = []
    for n in range(0, number_of_frames):
        frame_sample = sig[i:(samples_per_frame*x)]
        i = samples_per_frame*x
        x += 1
        frames.append(frame_sample)


    mfcc_frames = []
    for frame in frames:
        mel = mfcc(frame, rate, win_length)
        mfcc_frames.append(mel)

    return mfcc_frames
"""

def get_features(file_name):
    (rate, sig) = wavfile.read(file_name)

    with contextlib.closing(wave.open(file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # Frame our signal into 20 frames with 50% overlap
    number_of_frames = 40
    frame_len = len(sig) / (number_of_frames*(.5) + .5)
    print(sig.shape)
    frames = framesig(sig, frame_len, frame_len * .5)

    # A list of 20 frequency lists for each frame. 6 frequency bands with the average energy of each
    features = []
    band0 = []
    band1 = []
    band2 = []
    band3 = []
    band4 = []
    band5 = []
    for frame in frames:
        spectrum, freqs, t, img = specgram(frame, Fs=rate)
        i = 0
        bands = []
        for freq in freqs:
            if freq <= 333.3:
                band0.extend(spectrum[i])
            elif freq > 333.3 and freq <= 666.7:
                band1.extend(spectrum[i])
            elif freq > 666.7 and freq <= 1333.3:
                band2.extend(spectrum[i])
            elif freq > 1333.3 and freq <= 2333.3:
                band3.extend(spectrum[i])
            elif freq > 2333.3 and freq <= 4000:
                band4.extend(spectrum[i])
            
            i += 1
        bands.append(sum(band0) / len(band0))
        bands.append(sum(band1) / len(band1))
        bands.append(sum(band2) / len(band2))
        bands.append(sum(band3) / len(band3))
        bands.append(sum(band4) / len(band4))
        features.append(bands)

    values = []
    for feature in features:
        for f in feature:
            values.append(f)

    return values

"""

def get_mel(file_name):
    (rate, sig) = wavfile.read(file_name)

    with contextlib.closing(wave.open(file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    number_of_frames = 40
    frame_len = len(sig) / (number_of_frames * (.5) + .5)
    step = frame_len * .5
    frames = framesig(sig, frame_len, step)
    win_length = (duration * 1000) / number_of_frames

    mel_values = []
    for frame in frames:
        mel_values.append(mfcc(frame, rate, win_length))

    values = []
    for v in mel_values:
        for coefs in v:
            for coef in coefs:
                values.append(coef)
    return values
"""

# Get label associated with this file
def get_label(filename):
    """
    head, tail = ntpath.split(filename)
    start = tail.index('-')
    tail = tail[(start+1):]
    end = tail.index('-')
    fname = tail[0:end]
    """
    head, tail = ntpath.split(filename)
    name = str(tail)
    label, rest1, rest2 = name.split('_')
    #print(label)
    return label


#get_label('/home/aiswarya/SNN_works/my_code/spoken-digit-dataset/free-spoken-digit-dataset-master/recordings/0_jackson_0.wav')