import numpy as np
import torch

from signal_utils import generate_synthetic_features, Signal
import train_funcs, eval_funcs

def main():

    # Create synthetic dataset
    single_band = False
    num_elec = 16
    timesteps = 1000
    fs = 500
    t = np.arange(1000) / fs

    # Create across feature engineered bands or restrict to one band
    if not single_band:
        data = generate_synthetic_features(num_elec, timesteps, fs)
    else: # Restrict to one band
        f = 50 # Sitting in the middle of low-gamma
        num_channels = num_elec * 4 # Since we are not feeding each band in
        raw_data = np.sin(2 * np.pi * f * t) + 0.5 * np.random.randn(num_channels, len(t))

        # Call Signal class, normalize and bandpass
        sig = Signal(
            raw_data, 
            fs=600, 
            name="SynthSignal"
        )
        sig.normalize()
        sig.bandpass_filter(band="lower_gamma")
        data = sig.data

    # Train the TCN to predict the next timestep
    check_latent = False
    x, tcn_trained = train_funcs.train_tcn(data)
    torch.save(tcn_trained.state_dict(), "tcn_frozen.pth")
    if check_latent:
        eval_funcs.check_latent_feature(x, tcn_trained)
        eval_funcs.check_heatmap(x, tcn_trained)

if __name__ == "__main__":
    main()
