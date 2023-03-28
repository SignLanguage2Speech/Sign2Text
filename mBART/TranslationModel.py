from transformers import MBartForConditionalGeneration
import torch.nn as nn
import torch

class TranslationModel(nn.Module):
    def __init__(self, model_path, device):
        super(TranslationModel, self).__init__()
        
        self.device = device
        self.mbart = MBartForConditionalGeneration.from_pretrained(model_path).to(self.device)

    def generate(self, x):
        label_ids = self.mbart.generate(
            inputs_embeds = x, 
            decoder_inputs_embeds = x,
            num_beams = 4,
            early_stopping = True,
            repetition_penalty = 3.0
        )
        return label_ids

    def forward(self, x):
        logits = self.mbart(inputs_embeds = x, decoder_inputs_embeds = x).logits
        return logits