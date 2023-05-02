import torch

def load_model_from_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model