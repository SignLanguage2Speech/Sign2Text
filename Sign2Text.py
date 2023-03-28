import torch
import os
from torchsummary import summary
from torchvision.ops import MLP
from torch import nn
from VL_mapper.get_VL_mapper import get_VL_mapper
from S3D.utils.get_visual_model import get_visual_model
from mBART.TranslationModel import TranslationModel
from mBART.get_tokenizer import get_tokenizer
from Phoenix.PHOENIXDataset import PhoenixDataset
import pandas as pd

class Sign2Text(torch.nn.Module):
    def __init__(self, visual_model_vocab_size, visual_model_checkpoint, n_visual_features, mbart_path, device):
        super(Sign2Text, self).__init__()

        self.device = device
        self.visual_model = get_visual_model(visual_model_vocab_size, visual_model_checkpoint, self.device)
        self.VL_mapper = get_VL_mapper(n_visual_features, device)
        self.language_model = TranslationModel(mbart_path, device)
        self.tokenizer = get_tokenizer(mbart_path)

    def forward(self, x):
        reps, probs = self.visual_model(x)
        visual_features_permute = reps.permute(0,2,1)
        gloss_representations = self.VL_mapper(visual_features_permute)
        out = self.language_model(gloss_representations)
        return out, probs

    def predict(self, x, skip_special_tokens = True):
        reps, _ = self.visual_model(x)
        visual_features_permute = reps.permute(0,2,1)
        gloss_representations = self.VL_mapper(visual_features_permute)
        label_ids = self.language_model.generate(gloss_representations)
        preds = self.tokenizer.batch_decode(label_ids, skip_special_tokens=skip_special_tokens)
        return preds