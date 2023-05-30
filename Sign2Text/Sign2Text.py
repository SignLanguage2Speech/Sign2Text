import torch
import os
from torchsummary import summary
from torchvision.ops import MLP
from torch import nn
from model.Sign2Text.VisualEncoder.VisualEncoder import VisualEncoder
from model.Sign2Text.VL_mapper.get_VL_mapper import get_VL_mapper
from model.Sign2Text.GL_mapper.get_GL_mapper import get_GL_mapper
from model.Sign2Text.mBART.TranslationModel import TranslationModel
from model.Sign2Text.mBART.get_tokenizer import get_tokenizer

class Sign2Text(torch.nn.Module):
    def __init__(self, Sign2Text_cfg, VisualEncoder_cfg):
        super(Sign2Text, self).__init__()

        self.device = Sign2Text_cfg.device
        self.visual_encoder = VisualEncoder(VisualEncoder_cfg)
        self.language_model = TranslationModel(Sign2Text_cfg)
        self.VL_mapper = get_VL_mapper(Sign2Text_cfg)

    def get_language_params(self):
        return self.language_model.parameters()

    def get_visual_params(self):
        return list(filter(lambda p: p.requires_grad, self.visual_encoder.parameters())) \
        + list(self.VL_mapper.parameters())

    def get_params(self, CFG):
        return [{'params':self.get_language_params(), 'lr': CFG.init_lr_language_model},
            {'params':self.get_visual_params(), 'lr': CFG.init_lr_visual_model}]

    def forward(self, x, trg, ipt_len):
        probs, reps = self.visual_encoder(x, ipt_len)
        gloss_representations = self.VL_mapper(reps)
        out, loss = self.language_model(gloss_representations, trg, ipt_len)
        return out, probs, loss

    def predict(self, x, ipt_len, skip_special_tokens = True):
        probs, reps = self.visual_encoder(x, ipt_len)
        gloss_representations = self.VL_mapper(reps)
        preds = self.language_model.generate(
            gloss_representations, 
            ipt_len,
            skip_special_tokens = skip_special_tokens)
        return preds