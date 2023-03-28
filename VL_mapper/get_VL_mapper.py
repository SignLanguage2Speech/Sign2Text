from torch import nn

def get_VL_mapper(n_visual_features, device):
    return  nn.Sequential(
                nn.Linear(n_visual_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            ).to(device)