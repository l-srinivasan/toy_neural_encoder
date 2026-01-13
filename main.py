import os
import numpy as np
import pandas as pd

from signal_utils import generate_synthetic_features, Signal
from models import ThreeLayerTCN
from train import train_tcn

def main():

    # Create synthetic dataset
    single_band = False
    num_elec = 16
    timesteps = 1000
    fs = 500
    t = np.arange(1000) / fs


    if not single_band:
        data = generate_synthetic_features(num_elec, timesteps, fs)
    else: # Restrict to one band
        f = 50 # Sitting in the middle of low-gamma
        raw_data = np.sin(2 * np.pi * f * t) + 0.5 * np.random.randn(num_elec, len(t))

        # Call Signal class, normalize and bandpass
        sig = Signal(
            raw_data, 
            fs=600, 
            name="TestSignal"
        )
        sig.normalize()
        sig.bandpass_filter(band="lower_gamma")
        data = sig.data

    tcn = train_tcn(data)

if __name__ == "__main__":
    main()
