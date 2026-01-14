import numpy as np
from scipy.signal import lfilter

class Signal():
    def __init__(self, data, fs=600, name=None):
        self.data = np.array(data, dtype=float)
        self.fs = fs
        self.name = name
    
    def normalize(self, epsilon=1e-8):
        # Channel wise mean and std
        mu = np.mean(self.data, axis=1, keepdims=True)
        std = np.std(self.data, axis=1, keepdims=True)

        # Vectorized normalization, safe for std=0
        self.data = (self.data - mu) / (std + epsilon)

    def bandpass_filter(self, band="lower_gamma"):

        # Identify bands of interest
        bands = {
            "low": (1, 8),
            "mu": (8, 13),
            "beta": (13, 30),
            "lower_gamma": (30, 70),
            "high_gamma": (70, 200)
        }
        low, high = bands[band]

        # Apply Fourier transform
        n = self.data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1/self.fs)
        fft_data = np.fft.rfft(self.data, axis=1)

        # Mask and apply inverse transform
        mask = (freqs >= low) & (freqs <= high)
        fft_data *= mask
        self.data = np.fft.irfft(fft_data, n=n, axis=1)


def generate_synthetic_features(
    num_elec=16,
    timesteps=1000,
    fs=500,
    noise_type='gaussian',
    seed=None
):

    if seed is not None:
        np.random.seed(seed)
    t = np.arange(timesteps) / fs

    # Frequency ranges for each channel
    bands = [
        (30, 80),   # Primary envelope
        (15, 30),   # Beta
        (70, 120),  # High-gamma
        (1, 8)      # Low frequency
    ]

    n_channels = num_elec * len(bands)
    x = np.zeros((n_channels, timesteps))

    # Inject independent noise to each channel
    for elec in range(num_elec):
        for band_idx, (f_low, f_high) in enumerate(bands):
        
            freq = np.random.uniform(f_low, f_high)
            sine_wave = np.sin(2 * np.pi * freq * t)

            # Noise type
            if noise_type == 'gaussian':
                noise = 0.3 * np.random.randn(timesteps)
            elif noise_type == 'smooth':
                raw_noise = 0.3 * np.random.randn(timesteps)
                kernel = np.ones(5)/5
                noise = np.convolve(raw_noise, kernel, mode='same')
            else:
                raise ValueError("noise_type must be 'gaussian' or 'smooth'")

            # Inject sine wave with noise
            idx = elec*len(bands) + band_idx
            x[idx] = sine_wave + noise

    # Normalize
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return x

def event_driven_changes():

    """
    Docstring for event_driven_changes

    ** Could introduce events to test latent feature variation over time
    """
    pass
    