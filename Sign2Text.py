import torch
import os
from torchsummary import summary
from torchvision.ops import MLP
from torch import nn
from VL_mapper.get_VL_mapper import get_VL_mapper
from GL_mapper.get_GL_mapper import get_GL_mapper
from S3D.utils.get_visual_model import get_visual_model
from mBART.TranslationModel import TranslationModel
from Phoenix.PHOENIXDataset import PhoenixDataset
import pandas as pd

class Sign2Text(torch.nn.Module):
    
    def __init__(self, CFG):
        super(Sign2Text, self).__init__()

        self.device = CFG.device
        self.visual_model = get_visual_model(CFG.visual_model_vocab_size, CFG.visual_model_checkpoint, CFG.device)
        self.VL_mapper = get_VL_mapper(CFG.n_visual_features, CFG.device)
        self.GL_mapper = get_GL_mapper(CFG.visual_model_vocab_size, CFG.device)
        self.language_model = TranslationModel(CFG.mbart_path, CFG.beam_width, CFG.max_seq_length, CFG.length_penalty, CFG.device)
        self.use_GL_mapper = CFG.use_GL_mapper

    def forward(self, x):
        if torch.isnan(x).any():
            print("INPUT FAILURE")
        reps, probs = self.visual_model(x)
        if torch.isnan(reps).any() or torch.isnan(probs).any():
            print("VISUAL FAILURE")
        visual_features_permute = reps.permute(0,2,1)
        visual_language_features = self.VL_mapper(visual_features_permute)
        if torch.isnan(visual_language_features).any():
            print("VL_mapper FAILURE")
        if self.use_GL_mapper:
            gloss_language_features = self.GL_mapper(probs)
            if torch.isnan(gloss_language_features).any():
                print("GL_mapper FAILURE")
            out = self.language_model(visual_language_features, gloss_language_features)
            if torch.isnan(out).any():
                print("language_model FAILURE")
        else:
            out = self.language_model(visual_language_features)
            if torch.isnan(out).any():
                print("language_model FAILURE")
        return out, probs

    def predict(self, x, skip_special_tokens = True):
        reps, probs = self.visual_model(x)
        visual_features_permute = reps.permute(0,2,1)
        visual_language_features = self.VL_mapper(visual_features_permute)
        if self.use_GL_mapper:
            gloss_language_features = self.GL_mapper(probs)
            preds = self.language_model.generate(visual_language_features, gloss_language_features, skip_special_tokens)
        else:
            preds = self.language_model.generate(visual_language_features, skip_special_tokens = skip_special_tokens)
        return preds