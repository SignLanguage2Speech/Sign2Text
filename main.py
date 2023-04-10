import os
import torch
from Sign2Text import Sign2Text
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from Trainer.trainer import train
from Phoenix.PHOENIXDataset import PhoenixDataset
from Phoenix.PHOENIXDataset import collator
import pandas as pd
from torch.utils.data import DataLoader
from utils.compute_metrics import compute_metrics

class cfg:
    def __init__(self):

        ### paths ###
        self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
        self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
        self.save_path = os.path.join('/work3/s204138/bach-models', 'PHOENIX_trained_models')
        self.mbart_path = '/work3/s200925/mBART/checkpoints/checkpoint-70960'
        self.visual_model_checkpoint = os.path.join(self.save_path, '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc')
        
        ### dimensions ###
        self.visual_model_vocab_size = 1088
        self.n_visual_features = 512

        ### model params ###
        self.use_GL_mapper = True
        self.beam_width = 4
        self.max_seq_length = 150
        self.length_penalty = 1.0

        ### training params ###
        self.verbose = True
        self.verbose_batches = True
        self.num_workers = 8
        self.epochs = 100
        self.train_batch_size = 1
        self.val_batch_size = 1
        self.test_batch_size = 1
        self.init_lr = 0.00001
        self.decay_start_factor = 1.0
        self.decay_end_factor = 0.01
        self.ce_label_smoothing = 0.2
        
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    ### get config ###
    CFG = cfg()

    ### dataset ###
    if CFG.verbose:
        print("\n" + "-"*20 + "Preparing Data" + "-"*20)
    train_df = pd.read_csv(os.path.join(CFG.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
    val_df = pd.read_csv(os.path.join(CFG.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
    test_df = pd.read_csv(os.path.join(CFG.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')

    PhoenixTrain = PhoenixDataset(train_df, CFG.phoenix_videos, vocab_size=CFG.visual_model_vocab_size, split='train')
    PhoenixVal = PhoenixDataset(val_df, CFG.phoenix_videos, vocab_size=CFG.visual_model_vocab_size, split='dev')
    PhoenixTest = PhoenixDataset(test_df, CFG.phoenix_videos, vocab_size=CFG.visual_model_vocab_size, split='test')
    
    dataloaderTrain = DataLoader(PhoenixTrain, batch_size=CFG.train_batch_size, 
                                   shuffle=True,
                                   num_workers=CFG.num_workers,
                                   collate_fn = collator)
    dataloaderVal = DataLoader(PhoenixVal, batch_size=CFG.val_batch_size, 
                                   shuffle=False,
                                   num_workers=CFG.num_workers)
    dataloaderTest = DataLoader(PhoenixTest, batch_size=CFG.test_batch_size, 
                                   shuffle=False,
                                   num_workers=CFG.num_workers)
    
    ### initialize model ###
    if CFG.verbose:
        print("\n" + "-"*20 + "Preparing Model" + "-"*20)
    model = Sign2Text(CFG)

    ### train model ###
    if CFG.verbose:
        print("\n" + "-"*20 + "Preparing Training" + "-"*20)
    train(model, dataloaderTrain, dataloaderVal, CFG)

    ### evaluate model on test ###
    print("\n" + "-"*20 + "Preparing Evaluation" + "-"*20)
    print(compute_metrics(model, dataloaderTest, CFG))


if __name__ == '__main__':
    main()