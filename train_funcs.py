import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ThreeLayerTCN, TimeSeriesTransformer

class TCNPredictNext(nn.Module):
    def __init__(self, tcn, out_channels):
        super().__init__()

        self.tcn = tcn
        self.prediction_head = nn.Linear(out_channels, out_channels)

    def forward(self, x):

        features = self.tcn(x)
        preds = self.prediction_head(features)
        return preds # In [Batches, Timesteps, Channels=out_channels]

def train_tcn(x):

    # Prepare input data
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0) # Gives us [Batches, Channels, Timesteps] for the TCN

    # Include/exclude next timestep
    x_input = x[:,:,:-1]
    x_target = x[:,:,1:]

    # Instantiate model
    tcn = ThreeLayerTCN(64, 128, 128)
    model = TCNPredictNext(tcn, 128)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # Iterate and train
    for epoch in range(500):
        optimizer.zero_grad()

        # Forward pass
        preds = model(x_input)
        with torch.no_grad():
            target_features = tcn(x_target)
        loss = criterion(target_features, preds)

        # Calculate loss and update gradients
        loss.backward()
        optimizer.step()

        if (epoch % 50) == 0:
            print(f"Epoch {epoch} ; Loss {loss.item():.4f}")

    # Freeze weights
    for param in tcn.parameters():
        param.requires_grad = False
    return x, tcn
    
class TransformerPredictNext(nn.Module):
    def __init__(self, tcn_trained, d_model):
        super().__init__()

        self.tcn = tcn_trained
        self.tcn.eval() # Freeze the TCN
        for param in self.tcn.parameters():
            param.requires_grad=False

        self.transformer = TimeSeriesTransformer(tcn_trained, d_model=d_model)
        self.prediction_head = nn.Linear(d_model, d_model)

    def forward(self, x):

        with torch.no_grad():
            z = self.tcn(x)
        
        features = self.transformer(z)
        preds = self.prediction_head(features)
        return preds
    
def train_transformer(x, tcn_trained, device="cpu"):

    # Prep input and target sequences
    x_input = x[:,:,:-1].to(device) # Assuming x is a tensor
    x_target = x[:,:,1:].to(device)

    # Instantiate model
    mod = TransformerPredictNext(tcn_trained, 128).to(device)
    optimizer = torch.optim.Adam(mod.parameters())
    criterion = nn.MSELoss()

    # Iterate and train
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad() # Restrict gradients from accum. across epochs
        
        # Generate target
        preds = mod(x_input)
        with torch.no_grad():
            target_latent = tcn_trained(x_target)

        # Calculate loss
        loss = criterion(preds, target_latent)
        loss.backward()
        optimizer.step()

        # Report loss every 10 epochs
        if ((epoch+1) % 10) == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():6f}")

    return mod