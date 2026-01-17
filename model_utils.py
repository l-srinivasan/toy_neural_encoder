import torch

from models import ThreeLayerTCN, TimeSeriesTransformer

def load_trained_tcn_weights(path):

    tcn = ThreeLayerTCN(64, 128, 128)
    state_dict = torch.load(path, weights_only=True)
    tcn.load_state_dict(state_dict)
    return tcn

def load_trained_tf_weights(path, tcn_trained):
    
    tf = TimeSeriesTransformer(tcn_trained, 128)
    model_dict = tf.state_dict()

    state_dict = torch.load(path, weights_only=True)
    state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
    
    tf.load_state_dict(state_dict)
    return tf