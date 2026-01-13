import torch
import torch.nn as nn

def eval_tcn(tcn, sig):
    
    x = torch.tensor(sig.data, dtype=torch.float32)
    x = x.unsqueeze(0) # Add batch dimension
    
    if x.shape[1] != tcn.conv.in_channels:
        x = x.transpose(1, 2)  # make it [B, C, T]

    tcn.eval()
    with torch.no_grad():
        print(">>> before forward()", flush=True)
        features = tcn(x)
        print(">>> after forward()", flush=True)

    return features.squeeze(0).cpu().numpy()
    


    

