import torch

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_b4 = checkpoint['val_b4']
    return model, optimizer, scheduler, epoch, train_losses, val_b4