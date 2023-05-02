import torch

def save_checkpoint(save_path, model, optimizer, scheduler, epoch, epoch_losses, val_b4):
  torch.save({'epoch' : epoch,
              'model_state_dict' : model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict(),
              'scheduler_state_dict' : scheduler.state_dict(),
              'train_losses' : epoch_losses,
              'val_b4' : val_b4,
              }, save_path)