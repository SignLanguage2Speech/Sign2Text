

def freeze_params(module):
    for _, p in module.named_parameters():
        p.requires_grad = False