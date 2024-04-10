#!/usr/bin/env python
# coding: utf-8

# Load packages
import os
import pandas as pd
import numpy as np
import patoolib
import pathlib
from tqdm import tqdm
from scipy import signal

# Read data
    # Source: https://www.kaggle.com/code/andersgb/nasa-bearing-dataset-outlier-detection

def read_dataset(data_dir, first_ts, colnames, rate=20480):
    all_dfs = []
    for file_counter, f in enumerate(tqdm(sorted(pathlib.Path(data_dir).iterdir()))):
        df = pd.read_csv(f, sep="\t", header=None, dtype=np.float32)#.rename(columns=colnames)
        ts = pd.to_datetime(f.name, format="%Y.%m.%d.%H.%M.%S")
        measurement_delta = (ts - first_ts).total_seconds()
        step_s = 1 / rate  # 20 kHz sampling -- but these are 1 second snapshots! So appears to be 20,480 Hz
        df["time"] = measurement_delta + np.arange(len(df)) * step_s
        df["measurement_id"] = file_counter
        df["measurement_id"] = df["measurement_id"].astype(np.uint32)
        if len(df) != rate:
            raise RuntimeError(f"Unexpected file length {len(df)} in {f}")
        all_dfs.append(df)
    df_out = pd.concat(all_dfs, ignore_index=True)
    df_out["counter"] = df_out.groupby('measurement_id').cumcount()
    df_out.columns = colnames
    return df_out


# Downsample the signal after applying an anti-aliasing filter.
    # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html
        # by default, order 8 Chebyshev type I filter is used if ftype is 'iir'. HOWEVER this results in all 'nan' values
        # 30 point FIR filter with Hamming window is used if ftype is ‘fir’.
    # Source: https://en.wikipedia.org/wiki/Anti-aliasing_filter

def down_sample(df, col, q):  
    arr = signal.decimate(np.array(df[col]), q, ftype='fir') 
    return arr
