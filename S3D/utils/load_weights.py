import os
import torch

def load_PHOENIX_weights(model, path='/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc', verbose=False):
    
    checkpoint = torch.load(path)
    weights = checkpoint['model_state_dict']
    sd = model.state_dict()
    
    for name, param in weights.items():
        
        if name in sd:
            if param.size() == sd[name].size():
                sd[name].copy_(param)
            else:
                if verbose:
                    print(f"Dimensions do not match...\n parameter: {name} has size {param.size()}\n Original size is: {sd[name].size()}")
        else:
            if verbose:
                print(f"Param {name} not in state dict")
            
