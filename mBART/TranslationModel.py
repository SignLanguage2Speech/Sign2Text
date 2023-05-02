from transformers import MBartForConditionalGeneration
from transformers import BeamSearchScorer
from mBART.get_tokenizer import get_tokenizer
import torch.nn as nn
import torch

class TranslationModel(nn.Module):
    def __init__(self,CFG):
        super(TranslationModel, self).__init__()
        
        self.device = CFG.device
        self.tokenizer = get_tokenizer(CFG.mbart_path)
        self.mbart = MBartForConditionalGeneration.from_pretrained(CFG.mbart_path).to(CFG.device) 
        self.beam_width = CFG.beam_width
        self.max_seq_length = CFG.max_seq_length
        self.length_penalty = CFG.length_penalty

    # def generate(self, visual_language_features, gloss_language_features = None, skip_special_tokens = True):
    #     logits = self.forward(visual_language_features, gloss_language_features)
    #     generated_ids = self.beam_search(logits)
    #     preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)
    #     return preds
    def generate(self, visual_language_features, gloss_language_features = None, skip_special_tokens = True):
        input_lengths = torch.tensor([visual_language_features.shape[1] for _ in range(visual_language_features.shape[0])])
        batch_size = visual_language_features.shape[0]
        kwargs = self.prepare_feature_inputs(visual_language_features, input_lengths)
        kwargs["decoder_input_ids"] = torch.ones([batch_size,1],dtype=torch.long, device=visual_language_features.device)*self.tokenizer.bos_token_id
        output_dict = self.mbart.generate(
            inputs_embeds=kwargs["inputs_embeds"], attention_mask=kwargs["attention_mask"],
            decoder_input_ids=kwargs["decoder_input_ids"],
            num_beams=self.beam_width, length_penalty=self.length_penalty, max_length=self.max_seq_length, 
            return_dict_in_generate=True)
        text = self.tokenizer.batch_decode(output_dict['sequences'], skip_special_tokens=skip_special_tokens)
        return text

    # def forward(self, visual_language_features, gloss_language_features = None):
    #     encoder_outputs = self.mbart.model.encoder(inputs_embeds=visual_language_features)
    #     if gloss_language_features is not None:
    #         decoder_outputs = self.mbart.model.decoder(
    #             inputs_embeds=gloss_language_features, 
    #             encoder_hidden_states=encoder_outputs.last_hidden_state)
    #     else:
    #         decoder_outputs = self.mbart.model.decoder(
    #             inputs_embeds=visual_language_features, 
    #             encoder_hidden_states=encoder_outputs.last_hidden_state)
    #     logits = self.mbart.lm_head(decoder_outputs.last_hidden_state)
    #     return logits

    def forward(self, visual_language_features, gloss_language_features = None):
        input_lengths = torch.tensor([visual_language_features[i].shape[0] for i in range(visual_language_features.shape[0])])
        batch_size = visual_language_features.shape[0]
        feature_dim = visual_language_features.shape[1]
        kwargs = self.prepare_feature_inputs(visual_language_features, input_lengths)
        kwargs["decoder_input_ids"] = torch.ones([batch_size,feature_dim],dtype=torch.long, device=visual_language_features.device)*self.tokenizer.pad_token_id
        kwargs["decoder_input_ids"][:,0] = self.tokenizer.bos_token_id
        out = self.mbart(**kwargs)
        return out.logits

    def prepare_feature_inputs(self, input_feature, input_lengths, gloss_embedding=None, gloss_lengths=None):
        suffix_len = 0
        suffix_embedding = None

        max_length = torch.max(input_lengths)+suffix_len
        inputs_embeds = []
        attention_mask = torch.zeros([input_feature.shape[0], max_length], dtype=torch.long, device=input_feature.device)

        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii] 
            valid_feature = feature[:valid_len,:]
            feature_w_suffix = valid_feature
            if feature_w_suffix.shape[0]<max_length:
                pad_len = max_length-feature_w_suffix.shape[0]
                padding = torch.zeros([pad_len, feature_w_suffix.shape[1]], 
                    dtype=feature_w_suffix.dtype, device=feature_w_suffix.device)
                padded_feature_w_suffix = torch.cat([feature_w_suffix, padding], dim=0) #t+2+pl,D
                inputs_embeds.append(padded_feature_w_suffix)
            else:
                inputs_embeds.append(feature_w_suffix)
            attention_mask[ii, :valid_len+suffix_len] = 1
        transformer_inputs = {
            'inputs_embeds': torch.stack(inputs_embeds, dim=0),
            'attention_mask': attention_mask,
            'input_ids' : None,
        }
        return transformer_inputs

    def beam_search(self, logits):

        beam_width = self.beam_width
        max_seq_length = self.max_seq_length
        length_penalty = self.length_penalty
        end_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        batch_size, sequence_length, vocab_size = logits.shape
        
        # Initialize the beams
        beams = [[([], 0.0)] for _ in range(batch_size)]
        completed_beams = [[] for _ in range(batch_size)]

        for step in range(sequence_length):
            candidates = []

            # Expand the beams
            for i in range(batch_size):
                for beam in beams[i]:
                    sequence, score = beam
                    last_token_id = sequence[-1] if sequence else None

                    # Compute the scores for each candidate token
                    scores = logits[i, step] - length_penalty * len(sequence)

                    # Exclude the end token if it has already been added to the sequence
                    if last_token_id == end_token_id:
                        scores[end_token_id] = float('-inf')

                    # Select the top-k candidates
                    top_k = torch.topk(scores, beam_width).indices.tolist()

                    for token_id in top_k:
                        candidates.append((i, sequence + [token_id], score + scores[token_id]))

            # Split the candidates into batches and sort and crop independently for each batch
            candidates_by_batch = [[] for _ in range(batch_size)]
            for candidate in candidates:
                candidates_by_batch[candidate[0]].append(candidate)

            beams = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                candidates_i = candidates_by_batch[i]
                if candidates_i:
                    candidates_i = sorted(candidates_i, key=lambda x: x[2], reverse=True)[:beam_width]
                    for _, sequence, score in candidates_i:
                        if sequence[-1] == end_token_id or len(sequence) == max_seq_length or len(sequence) == sequence_length:
                            # End the beam if the end token has been added or the sequence has reached its maximum length
                            completed_beams[i].append((sequence, score))
                        else:
                            beams[i].append((sequence, score))


        # Extract the best sequences
        outputs = torch.ones(batch_size, sequence_length, dtype=torch.long) * pad_token_id

        for i in range(batch_size):
            if completed_beams[i]:
                sequences, scores = zip(*completed_beams[i])
                best_sequence = sequences[scores.index(max(scores))]
                outputs[i, :len(best_sequence)] = torch.tensor(best_sequence)
        
        return outputs