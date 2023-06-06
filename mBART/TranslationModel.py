from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import BeamSearchScorer
from model.Sign2Text.mBART.get_tokenizer import get_tokenizer
import torch.nn as nn
import torch
import numpy as np
from model.Sign2Text.mBART.get_default_model import get_model_and_tokenizer, reduce_to_vocab
from model.Sign2Text.utils.freeze_params import freeze_params

class TranslationModel(nn.Module):
    def __init__(self,CFG):
        super(TranslationModel, self).__init__()
        
        self.device = CFG.device
        self.beam_width = CFG.beam_width
        self.max_seq_length = CFG.max_seq_length
        self.length_penalty = CFG.length_penalty
        
        if CFG.mbart_path is not None:
            print("Loading model from pretrained checkpoint!")
            self.tokenizer = get_tokenizer(CFG.mbart_path)
            self.mbart = MBartForConditionalGeneration.from_pretrained(CFG.mbart_path).to(CFG.device)
            freeze_params(self.mbart.model.shared)
        else:
            print("Loading model with cc25 initialization!")
            default_model, default_tokenizer = get_model_and_tokenizer(CFG)
            self.mbart, self.tokenizer = reduce_to_vocab(default_model, default_tokenizer, CFG) ### prune default mBART

    def generate(self, visual_language_features, input_lengths, skip_special_tokens = True):
        kwargs = self.prepare_feature_inputs(visual_language_features, input_lengths, generate = True)
        output_dict = self.mbart.generate(
            inputs_embeds=kwargs["inputs_embeds"],
            attention_mask=kwargs["attention_mask"],
            decoder_start_token_id = self.tokenizer.lang_code_to_id["de_DE"],
            num_beams=self.beam_width, 
            length_penalty=self.length_penalty, 
            max_length=self.max_seq_length, 
            return_dict_in_generate=True)
        text = self.tokenizer.batch_decode(output_dict['sequences'], skip_special_tokens=skip_special_tokens)
        return text

    def forward(self, visual_language_features, trg, input_lengths):
        kwargs = self.prepare_feature_inputs(visual_language_features, input_lengths, trg = trg)
        out = self.mbart(**kwargs)
        return out.logits, out.loss

    def prepare_feature_inputs(self, visual_language_features, input_lengths, generate = False, trg = None):
        batch_size, seq_length, hidden_size = visual_language_features.size()

        inputs_embeds = self.mbart.get_input_embeddings()(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0).repeat(batch_size,self.max_seq_length,1)
        attention_mask = torch.zeros([batch_size, self.max_seq_length], dtype=torch.long, device=visual_language_features.device)
        
        for i, feature in enumerate(visual_language_features):
            feature_len = int(np.floor(input_lengths[i]/4))
            # feature_len = len(feature)
            cropped_feature = feature[:feature_len]
            inputs_embeds[i,:feature_len,:] = cropped_feature
            attention_mask[i,:feature_len] = 1
        
        labels = None
        if not generate:
            labels = trg
            decoder_input_ids = None

        transformer_inputs =  {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'decoder_input_ids':decoder_input_ids,
            'input_ids' : None,
            'labels': labels,
        }
        return transformer_inputs