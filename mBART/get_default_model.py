from transformers import MBartForConditionalGeneration, MBartTokenizer
import pandas as pd
from torch import nn
import torch
from pathlib import Path
# from sentencepiece import sentencepiece_model_pb2 as spm
from typing import List

def get_model_and_tokenizer(CFG):
    tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25",  src_lang='de_DE', tgt_lang='de_DE')
    model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    return model, tokenizer


def reduce_to_vocab(model, tokenizer, CFG):
    phoenix_train_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv'
    df_train = pd.read_csv(phoenix_train_path, delimiter ='|')
    vocab = get_data_for_tokenizer(CFG.vocab_data, df_train)
    new_tokenizer, new_vocab_ids = get_new_tokenizer(tokenizer, vocab, CFG)
    pruned_model = prune_model_embeddings(model, new_tokenizer, new_vocab_ids, CFG)
    return pruned_model, new_tokenizer

import pdb
def get_data_for_tokenizer(data_path, df ):
    
    german_words = []
    file = open(data_path,'r',encoding='latin1')
    for line in file.readlines():
        german_words.append(line.replace("\n","").lower())
    # add train words and glosses to vocab
    words = [word.lower() for sent in df['translation'] for word in sent.split(' ')]
    glosses = [gloss.lower() for gloss_sent in df['orth'] for gloss in gloss_sent.split(' ')]
    german_words = list(set(german_words + words + glosses))
    return german_words

def prune_model_embeddings(model: MBartForConditionalGeneration, tokenizer: MBartTokenizer, vocab_ids, CFG):
    # vocab_ids = sorted(list(tokenizer.get_vocab().values()))
    # print(vocab_ids)
    #print(len(vocab_ids))
    trimmed_model = model
    changed_weights = {'final_logits_bias' : model.final_logits_bias[:, vocab_ids],
                        'model.shared.weight' : model.model.shared.weight.detach().numpy()[vocab_ids,:], 
                        'model.encoder.embed_tokens.weight' : model.model.encoder.embed_tokens.weight.detach().numpy()[vocab_ids, :], 
                        'model.decoder.embed_tokens.weight' : model.model.decoder.embed_tokens.weight.detach().numpy()[vocab_ids, :], 
                        'lm_head.weight' : model.lm_head.weight.detach().numpy()[vocab_ids, :]}

    # copy unchanged params over from the old model
    for param in model.state_dict().keys():
        if param in changed_weights.keys():
            continue
        trimmed_model.state_dict()[param].copy_(model.state_dict()[param])
    
    # set trimmed params
    if 'final_logits_bias' in model.state_dict():
        trimmed_model.final_logits_bias = changed_weights['final_logits_bias']

    prunedEmbeddingMatrix = torch.nn.Embedding.from_pretrained(torch.Tensor(changed_weights['model.shared.weight']), 
                                                            freeze=True, 
                                                            padding_idx=tokenizer.pad_token_id)

    trimmed_model.set_input_embeddings(prunedEmbeddingMatrix)

    if 'lm_head' in changed_weights.keys():
        prunedLMHeadMatrix = torch.Tensor(changed_weights['lm_head.weight'])
        _ = trimmed_model.lm_head.weight.data.copy_(prunedLMHeadMatrix)

    trimmed_model.tie_weights()
    """
    trimmed_model.generation_config.repetition_penalty = CFG.repetition_penalty
    trimmed_model.generation_config.early_stopping = CFG.early_stopping
    trimmed_model.generation_config.num_beams = CFG.num_beams
    trimmed_model.generation_config.length_penalty = CFG.length_penalty
    trimmed_model.config.dropout = CFG.dropout
    trimmed_model.config.attention_dropout = CFG.attention_dropout
    trimmed_model.config.classifier_dropout = CFG.classifier_dropout
    """
    trimmed_model.generation_config.decoder_start_token_id = tokenizer.lang_code_to_id[CFG.src_lang]
    trimmed_model.config.vocab_size = len(tokenizer.get_vocab())
    return trimmed_model



def get_new_vocab(tokenizer, data: List[str]):
    new_vocab = set()
    new_vocab_ids = None

    for sent in data:
        tokenized = tokenizer(sent.lower(), add_special_tokens=False, return_attention_mask=True)
        for e in tokenized['input_ids']:
            new_vocab.add(tokenizer.convert_ids_to_tokens(e))
    
    # add special tokens to the new vocab
    new_vocab.update(tokenizer.all_special_tokens)
    new_vocab.update(tokenizer.additional_special_tokens)

    # add input ids from new vocab
    new_vocab_ids = sorted(tokenizer.convert_tokens_to_ids(new_vocab))
    return new_vocab, new_vocab_ids



def get_new_tokenizer(tokenizer: MBartTokenizer, data: List[str], CFG):

    new_vocab, new_vocab_ids = get_new_vocab(tokenizer, data)
    
    # save old tokenizer configs
    tokenizer.save_pretrained('tokenizer')

    # load vocab into spm
    # m = spm.ModelProto()
    m = None
    m.ParseFromString(open('tokenizer/sentencepiece.bpe.model', 'rb').read())
    N = len(m.pieces)

    # check words that are in the new vocab
    for i in range(N):
        e = m.pieces[i]
        if e.piece in new_vocab:
            m.pieces.append(e)

    # delete oov words
    del m.pieces[:N]

    # save new model
    with open('tokenizer/sentencepiece.bpe.model', 'wb') as f:
        f.write(m.SerializeToString())

    new_tokenizer = MBartTokenizer.from_pretrained('tokenizer',  src_lang=CFG.trg_lang, tgt_lang=CFG.trg_lang)
    new_special_tokens = new_tokenizer.additional_special_tokens + [CFG.src_lang]
    new_tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
    new_tokenizer.lang_code_to_id[CFG.src_lang] = new_tokenizer.convert_tokens_to_ids([CFG.src_lang])[0]
    new_tokenizer.src_lang = CFG.src_lang

    new_vocab_ids.append(new_tokenizer.convert_tokens_to_ids([CFG.src_lang])[0])
    
    return new_tokenizer, new_vocab_ids