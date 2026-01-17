import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeLayerTCN(nn.Module):
    def __init__(
            self,
            in_channels, # Electrodes x 
            hidden_channels,
            out_channels, # Should be 128 for the Transformer
            kernel_size=3,
            dilations = [1,2,4]
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size,
            padding = ((kernel_size - 1) * dilations[0]) // 2,
            dilation = dilations[0]
        )

        self.conv2 = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size,
            padding = ((kernel_size - 1) * dilations[1]) // 2,
            dilation = dilations[1]
        )

        self.conv3 = nn.Conv1d(
            hidden_channels,
            out_channels,
            kernel_size,
            padding = ((kernel_size - 1) * dilations[2]) // 2,
            dilation = dilations[2]
        )

        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.ln3 = nn.LayerNorm(out_channels)

        self.act = nn.ReLU()
    
    def forward(self, x):
        
        # Input dimension for x: [Batches, Channels, Timesteps]
        x = self.conv1(x)
        x = x.transpose(1,2)
        x = self.act(self.ln1(x)) # LayerNorm expects [B,T,C] here
        x = x.transpose(1,2)

        x = self.conv2(x)
        x = x.transpose(1,2)
        x = self.act(self.ln2(x)) # LayerNorm expects [B,T,C] here
        x = x.transpose(1,2)

        x = self.conv3(x)
        x = x.transpose(1,2)
        x = self.act(self.ln3(x)) # LayerNorm expects [B,T,C] here

        return x # In [Batches, Timesteps, out_channels]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
        trained_tcn, 
        d_model=128, 
        timesteps=1000,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()

        self.tcn = trained_tcn
        self.tcn.eval()
        for param in self.tcn.parameters():
            param.requires_grad = False # Prevent from further learning

        self.d_model = d_model
        self.pos_embedding = nn.Embedding(timesteps, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder=nn.TransformerEncoder(encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, z):

        # Verify shape
        B, T, C = z.shape
        assert z.shape[2] == self.d_model, f"Expected latent to have shape [B,T,C], but channel dimension has length {C}"

        # Create positional embedding
        pos = torch.arange(T)
        pos = pos.unsqueeze(0).expand(B,T) # Repeat the row B times
        embed_from_idx = self.pos_embedding(pos)

        # Append to latent for transformer
        z = z + embed_from_idx
        encoding = self.transformer_encoder(z)
        return encoding