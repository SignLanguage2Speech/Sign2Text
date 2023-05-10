import torch
from model.Sign2Text.configs.VisualEncoder_config import VisualEncoder_cfg
from model.Sign2Text.configs.Sign2Text_config import Sign2Text_cfg
from model.Sign2Text.configs.Training_config import Training_cfg
from model.Sign2Text.Sign2Text.Sign2Text import Sign2Text as s2t

def load_model_from_checkpoint(path, model=None, train=False, device='cpu'):
    if model is None:
        VE_CFG = VisualEncoder_cfg()
        S2T_CFG = Sign2Text_cfg()
        model = s2t(S2T_CFG, VE_CFG).to(torch.device(device)) # ! to(device) ? => only if we have a pc with NVIDIA GPU to test on
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    if not train:
        model.eval()
        model.zero_grad()
    return model