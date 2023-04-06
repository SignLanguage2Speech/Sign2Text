from torch import nn

def get_GL_mapper(n_classes:int, device):
    return  nn.Sequential(
                nn.Linear(n_classes, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            ).to(device)