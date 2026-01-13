import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np

from signal_utils import generate_synthetic_features, Signal
from models import TemporalConvNet, TimeSeriesTransformer
from models_2 import ThreeLayerTCN
from eval import eval_tcn

def main():

    # Create synthetic dataset
    all_bands="True"
    if all_bands=="True":
        data = generate_synthetic_features()
    else:
        # Build using class
        num_elec = 16
        fs = 500
        t = np.arange(1000) / fs
        raw_data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(16, len(t))

        # Call Signal class, normalize and bandpass
        sig = Signal(
            raw_data, 
            fs=600, 
            name="TestSignal"
        )
        sig.normalize()
        sig.bandpass_filter(band="high_gamma")
        data = sig.data

    # Temporal convolution
    tcn = ThreeLayerTCN(64, 128, 128, 3)
    print(tcn)
    return

    # Transformer
    trans = TimeSeriesTransformer(
        features.shape[1],
        num_heads = 8,
        num_layers = 8
    )
    print(trans)

if __name__ == "__main__":
    main()
