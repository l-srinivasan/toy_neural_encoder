import numpy as np
import torch

from signal_utils import generate_synthetic_features, Signal
import model_utils, train_funcs, eval_funcs

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
        num_channels = num_elec * 4 # Scale by 4 since we aren't feeding in bands as features
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
    load_tcn = True
    tcn_path =  "saved_params/tcn_frozen.pth"
    x_path = "saved_params/x_tensor.pth"

    if load_tcn:
        tcn_trained = model_utils.load_trained_tcn_weights(tcn_path)
        x = torch.load(x_path, weights_only=True)

    else:
        x, tcn_trained = train_funcs.train_tcn(data)
        torch.save(tcn_trained.state_dict(), tcn_path)
        torch.save(x, x_path)

        check_latent = False
        if check_latent:
            eval_funcs.check_latent_feature(x, tcn_trained)
            eval_funcs.check_heatmap(x, tcn_trained)

    # Train the Transformer to predict the next latent
    load_tf = True
    tf_path = "saved_params/tf_frozen.pth"

    if load_tf:
        tf_trained = model_utils.load_trained_tf_weights(tf_path, tcn_trained)

    else:
        tf_trained = train_funcs.train_transformer(x, tcn_trained)
        torch.save(tf_trained.state_dict(), tf_path)

    # Generate latent encoded state
    latent = tf_trained(x) # latent has shape [B, T, 128]; we do not want to compress temporal information

if __name__ == "__main__":
    main()
