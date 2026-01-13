import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

# Limit PyTorch threads to prevent deadlocks, disable problematic backends
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = True

class Dummy(nn.Module):
    def forward(self, x):
        return x
    
class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dil_size):
        super().__init__()
    
        padding = (kernel_size-1) // (2*dil_size)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dil_size,
            padding=padding
        )

        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        print(f"Before Conv1d: {x.shape}")
        x = self.conv(x)
        x = x.transpose(1,2) # Transpose for LayerNorm
        x = self.ln(x)
        x = self.act(x)
        return x
    
"""
Transformer learns
- Which timesteps matter more
- Which patterns repeat
- Which events contextualize others
- How to integrate long-range context
"""
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead = num_heads,
            dim_feedforward = 4 * d_model, # Standard
            batch_first=False
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.encoder(x)