import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from models import ThreeLayerTCN

def load_trained_tcn_weights(path):

    tcn = ThreeLayerTCN(64, 128, 128)
    state_dict = torch.load(path, weights_only=True)
    tcn.load_state_dict(state_dict)
    return tcn


def check_latent_feature(x, tcn_trained):

    """
    Docstring for check_latent_feature
    
    :param x: tensor used to train the tcn
    :param tcn_trained: tcn with frozen weights
    """

    x_latent = tcn_trained(x)
    x_latent = x_latent.squeeze(0)

    plt.plot(x_latent[0])
    plt.title("Visual Check: Latent Dimension 1 across Timesteps")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.show()

def check_heatmap(x, tcn_trained):

    """
    Docstring for check_heatmap
    
    :param x: tensor used to train the tcn
    :param tcn_trained: tcn with frozen weights
    """
    x_latent = tcn_trained(x)
    x_latent = x_latent.squeeze(0)

    plt.imshow(x_latent.T, aspect="auto")
    plt.title("Heatmap: Latent Representation across Timesteps")
    plt.xlabel("Time (ms)")
    plt.ylabel("Feature (n=128)")
    plt.show()