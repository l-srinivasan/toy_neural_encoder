import torch
import torch.nn as nn

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


