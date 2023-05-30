from torch import nn
import torch

def get_VL_mapper(CFG):
    return  nn.Sequential(
                nn.Linear(CFG.n_visual_features, 1024),
                nn.ReLU(),
                # nn.Dropout(p=CFG.VL_mapper_dropout),
                nn.Linear(1024, 1024),
            ).to(CFG.device)