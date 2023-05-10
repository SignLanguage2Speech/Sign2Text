from transformers import MBartForConditionalGeneration
from transformers import BeamSearchScorer
from mBART.get_tokenizer import get_tokenizer
import torch.nn as nn
import torch
import numpy as np

class TranslationModel(nn.Module):
    def __init__(self,CFG):
        super(TranslationModel, self).__init__()
        
        self.device = CFG.device
        self.beam_width = CFG.beam_width
        self.max_seq_length = CFG.max_seq_length
        self.length_penalty = CFG.length_penalty
        self.tokenizer = get_tokenizer(CFG.mbart_path)
        self.mbart = MBartForConditionalGeneration.from_pretrained(CFG.mbart_path).to(CFG.device) 
        self.mbart.config.dropout = CFG.mbart_dropout


    def generate(self, visual_language_features, input_lengths, skip_special_tokens = True):
        kwargs = self.prepare_feature_inputs(visual_language_features, input_lengths, generate = True)
        output_dict = self.mbart.generate(
            inputs_embeds=kwargs["inputs_embeds"], attention_mask=kwargs["attention_mask"],
            decoder_input_ids = kwargs["decoder_input_ids"],
            num_beams=self.beam_width, length_penalty=self.length_penalty, max_length=self.max_seq_length, 
            return_dict_in_generate=True)
        text = self.tokenizer.batch_decode(output_dict['sequences'], skip_special_tokens=skip_special_tokens)
        return text


    def forward(self, visual_language_features, trg, input_lengths):
        kwargs = self.prepare_feature_inputs(visual_language_features, input_lengths, trg = trg)
        out = self.mbart(**kwargs)
        return out.logits


    def prepare_feature_inputs(self, visual_language_features, input_lengths, generate = False, trg = None):
        batch_size, seq_length, hidden_size = visual_language_features.size()

        inputs_embeds = self.mbart.get_input_embeddings()(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0).repeat(batch_size,self.max_seq_length,1)
        attention_mask = torch.zeros([batch_size, self.max_seq_length], dtype=torch.long, device=visual_language_features.device)
        
        for i, feature in enumerate(visual_language_features):
            feature_len = int(np.floor(input_lengths[i]/4))
            cropped_feature = feature[:feature_len]
            inputs_embeds[i,:feature_len,:] = cropped_feature
            attention_mask[i,:feature_len] = 1
        
        decoder_input_ids = None
        labels = None
        if generate:
            decoder_input_ids = torch.ones([batch_size,1],dtype=torch.long, device=visual_language_features.device)*self.tokenizer.lang_code_to_id["de_DE"]
        else:
            labels = trg

        transformer_inputs =  {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'input_ids' : None,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }
        return transformer_inputs