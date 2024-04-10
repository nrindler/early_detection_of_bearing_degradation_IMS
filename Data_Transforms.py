#!/usr/bin/env python
# coding: utf-8

# # Transform Data: Time and Frequency Attributes

# Load packages
import os
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy.stats import kurtosis
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy.stats import entropy


# Fast Fourier Transform
    # Transform signal data to frequency domain
    # Source: https://github.com/DovileDo/BearingDegradationStageDetection/blob/main/src/data/TransformToFrequencyDomain.py
    # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftfreq.html
        # FFT returns Array of length n//2 + 1 containing the sample frequencies.

def FFT(arr, rate, q):
    X = np.empty((0, int((rate/q/2)+1)), float)

    for i in range(0, len(arr), int(rate/q)): # based on 1-second snapshots at given rate (20,480 Hz sampling frequency)
        # down-sampling by a factor of q
        #sample = signal.decimate(np.array(df[col])[i : i + 20480], 10, ftype='fir')
        x = np.abs(rfft(arr[i : i + int(rate/q)])) #sample
        X = np.append(X, np.array([x]), axis=0)
    return X


# ### Calculate Time Series attributes

# Transform to Time Domain
    # Source: https://github.com/DovileDo/BearingDegradationStageDetection/blob/main/src/data/TransformToFrequencyDomain.py
    # Source: https://github.com/DovileDo/BearingDegradationStageDetection/blob/main/src/data/Manual_labeling.py
        # Sahoo & Mohanty ("Multiclass Bearing Fault Classification Using Features Learned by a Deep Neural Network") 
        # identify the highest magnitude frequency of each observation, and the smoothed maximum acceleration 
            # (calculated by averaging the five highest absolute acceleration measurements in the time domain)
        # as predictive of bearing failure. These two attributes can be useful for manual labeling of bearing degradation stage.
    # Source: https://www.kaggle.com/code/furkancitil/nasa-bearing-dataset-rul-prediction

def time_features(arr, rate, q):

    zerocross = []
    ktosis = []
    rms = []
    peaks = []
    mean = []
    mean_abs = []
    std = []
    median = []
    skewness = []
    entrpy = []
    energy = []
    shapiro = []
    kl = []
    rkl = []
    crest = []
    abs_acc_5 = []
    max_freq = []
    max_abs = []
    shape = []
    impulse = []
    p2p = []
    
    for i in range(0, len(arr), int(rate/q)): # based on 1-second snapshots at given rate (20,480 Hz sampling frequency)
        # down-sampling by a factor of 10
        sample = arr[i : i + int(rate/q)]
        
        # smoothed maximum acceleration calculated by avg. 5 highest absolute acceleration measurements in the time domain.
        Hpositive = abs(sample)
        Hmax = np.argpartition(Hpositive, -5)[-5:]
        Hmean = np.mean(Hpositive[Hmax])
        abs_acc_5.append(Hmean)
        
        # highest magnitude frequency of each observation
        Hf = np.abs(rfft(sample))
        Hind = np.argpartition(Hf, -1)[-1:]# * 10
        max_freq.append(int(Hf[Hind]))
        
        # absolute max value
        abs_max = np.abs(sample).max()
        max_abs.append(abs_max)
        
        # zero crossing
        c = ((sample[:-1] * sample[1:]) < 0).sum() + (sample == 0).sum()
        zerocross.append(c)
        
        # kurtosis: distribution tail
        kur = kurtosis(sample)
        ktosis.append(kur)
        
        # Root Mean Square (RMS)
        rms_ = np.sqrt(np.mean(sample**2))
        rms.append(rms_)
        
        # Number of peaks
        pks, _ = find_peaks(sample)
        peaks.append(len(pks) / len(sample))
        
        # Mean
        mean.append(np.mean(sample))
        
        # Abs. Mean
        abs_mean_ = np.abs(sample).mean()
        mean_abs.append(abs_mean_) 
        
        # shape = rms / mean_abs
        shape.append(rms_ / abs_mean_)
        
        # impulse = max_abs / mean_abs
        impulse.append(abs_max / abs_mean_)
        
        # peak-to-peak = max_val - min_val
        p2p.append(np.max(sample) - np.min(sample))
        
        # Median
        abs_med_ = np.median(abs(sample))
        median.append(abs_med_)
        
        # std
        std.append(np.std(sample))
        
        # Skewness
        skewness.append(stats.skew(sample))
        
        # Entropy - extract shannon entropy (cut signals to 500 bins)
        entrpy.append(entropy(pd.cut(sample, 500).value_counts()))
        
        # Crest
        crest.append(
            np.max(np.abs(sample)) / np.sqrt(np.mean(np.square(sample))))
        
        # Energy
        energy.append(np.sum(np.abs(sample) ** 2))
        
        # Shapiro
        s, p = stats.shapiro(sample)
        shapiro.append(s)
        
        # KL
        x = np.linspace(min(sample), max(sample), 100)
        en = stats.entropy(
            stats.gaussian_kde(sample).evaluate(x),
            stats.norm.pdf(x, np.mean(sample), np.std(sample)),
        )
        kl.append(en)
        
        # Reverse KL
        x = np.linspace(min(sample), max(sample), 100)
        ren = stats.entropy(
            stats.norm.pdf(x, np.mean(sample), np.std(sample)),
            stats.gaussian_kde(sample).evaluate(x),
        )
        rkl.append(ren)
        
    df = pd.DataFrame(zerocross, columns=["zerocross"])
    df["ktosis"] = ktosis
    df["rms"] = rms
    df["peaks"] = peaks
    df["mean"] = mean
    df["mean_abs"] = mean_abs
    df["std"] = std
    df["median"] = median
    df["skewness"] = skewness
    df["crest"] = crest
    df["entrpy"] = entrpy
    df["energy"] = energy
    df["shapiro"] = shapiro
    df["kl"] = kl
    df["rkl"] = rkl
    df["abs_acc_5"] = abs_acc_5
    df["max_freq"] = max_freq
    df["max_abs"] = max_abs
    df["shape"] = shape
    df["impulse"] = impulse
    df["p2p"] = p2p
    
    mask = df["kl"] != np.inf
    df.loc[~mask, "kl"] = df.loc[mask, "kl"].max()

    mask = df["rkl"] != np.inf
    df.loc[~mask, "rkl"] = df.loc[mask, "rkl"].max()

    return df