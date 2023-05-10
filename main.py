import os
import torch
import pandas as pd
from Sign2Text.Sign2Text import Sign2Text
from configs.VisualEncoderConfig import cfg as VisualEncoder_cfg
from configs.Sign2Text_config import Sign2Text_cfg
from configs.Training_config import Training_cfg
from train_datasets.PHOENIXDataset import PhoenixDataset, collator, DataAugmentations
from torch.utils.data import DataLoader
from Trainer.trainer import train


def main():
    
    ### initialize configs and device ###
    VE_CFG = VisualEncoder_cfg()
    S2T_CFG = Sign2Text_cfg()
    T_CFG = Training_cfg()
    
    ### initialize data ###
    train_df = pd.read_csv(os.path.join(T_CFG.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')[:1]
    val_df = pd.read_csv(os.path.join(T_CFG.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
    test_df = pd.read_csv(os.path.join(T_CFG.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')

    ### initialize data ###
    PhoenixTrain = PhoenixDataset(train_df, T_CFG.phoenix_videos, vocab_size=VE_CFG.VOCAB_SIZE, split='train')
    PhoenixVal = PhoenixDataset(val_df, T_CFG.phoenix_videos, vocab_size=VE_CFG.VOCAB_SIZE, split='dev')
    PhoenixTest = PhoenixDataset(test_df, T_CFG.phoenix_videos, vocab_size=VE_CFG.VOCAB_SIZE, split='test')
    
    ### get dataloaders ###
    train_augmentations = DataAugmentations(split_type='train')
    val_augmentations = DataAugmentations(split_type='val')
    dataloader_train = DataLoader(
      PhoenixTrain, 
      collate_fn = lambda data: collator(data, train_augmentations), 
      batch_size=T_CFG.batch_size, 
      shuffle=True, num_workers=T_CFG.num_workers)
    dataloader_val = DataLoader(
      PhoenixVal, 
      collate_fn = lambda data: collator(data, val_augmentations), 
      batch_size=1, 
      shuffle=False,
      num_workers=T_CFG.num_workers)
    dataloader_test = DataLoader(
      PhoenixTest, 
      collate_fn = lambda data: collator(data, val_augmentations), 
      batch_size=1, 
      shuffle=False, 
      num_workers=T_CFG.num_workers) # TODO actually use this 🤡
    
    ### initialize model ###
    model = Sign2Text(S2T_CFG, VE_CFG).to(T_CFG.device)

    ### train model ###
    train(model, dataloader_train, dataloader_val, T_CFG)


if __name__ == '__main__':
  main()