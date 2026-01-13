import numpy as np

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

    def bandpass_filter(self, band="high_gamma"):

        # Identify bands of interest
        bands = {
            "low": (1, 8),
            "mu": (8, 13),
            "beta": (13, 30),
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


    