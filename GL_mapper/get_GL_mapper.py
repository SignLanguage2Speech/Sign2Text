from torch import nn

def get_GL_mapper(CFG):
    return  nn.Sequential(
                nn.Linear(CFG.n_classes, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            ).to(CFG.device)