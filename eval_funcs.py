import torch
import torch.nn as nn

import matplotlib.pyplot as plt

def check_latent_tcn(x, tcn_trained):

    x_latent = tcn_trained(x)
    x_latent = x_latent.squeeze(0)
    plt.plot(x_latent[0])
    plt.title("Visual Check: Latent Dimension 1 across Timesteps")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.show()