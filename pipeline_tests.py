print("Running...")
import torch
import torch.nn as nn
import time, os
import numpy as np
import pandas as pd
import pdb
from torch.utils.data import DataLoader
from torchmetrics.functional import word_error_rate
from torchaudio.models.decoder import ctc_decoder
print("General imports done")
from Sign2Text.Sign2Text import Sign2Text
from configs.VisualEncoderConfig import cfg as VE_cfg
from configs.Sign2Text_config import Sign2Text_cfg as S2T_cfg
from configs.Training_config import Training_cfg
from train_datasets.PHOENIXDataset import PhoenixDataset, collator, DataAugmentations
from train_datasets.preprocess_PHOENIX import getVocab, preprocess_df
from utils.compute_metrics import compute_metrics
print("All imports done..")

def tokens_to_sent(vocab, tokens):
  keys = list(vocab.keys())
  values = list(vocab.values())
  positions = [values.index(e) for e in tokens[tokens != 1086]]
  words = [keys[p] for p in positions]
  return ' '.join(words)

"""
From vision_model
"""
def validate(model, dataloader, criterion, decoder, CFG):

    ### setup validation metrics ###
    losses = []
    model.eval()
    start = time.time()
    word_error_rates = []
    secondary_word_error_rates = []

    ### iterature through dataloader ###
    for i, (ipt, vid_lens, trg, trg_len, _, _, _) in enumerate(dataloader): #(ipt, vid_lens, trg, trg_len, _, _)
        
        with torch.no_grad():
            ### get model output and calculate loss ###
            out, _ = model(ipt.to(CFG.device), vid_lens)
            x = out.permute(1, 0, 2)  
            ipt_len = torch.full(size=(1,), fill_value = out.size(1), dtype=torch.int32)
            loss = criterion(torch.log(x), 
                              trg, 
                              input_lengths=ipt_len,
                              target_lengths=trg_len) / out.size(0)
            
            ### save loss and get preds ###
            try:
                losses.append(loss.detach().cpu().item())
                out_d = decoder(out.cpu())
                preds = [p[0].tokens for p in out_d]
                pred_sents = [tokens_to_sent(CFG.gloss_vocab, s) for s in preds]
                ref_sents = tokens_to_sent(CFG.gloss_vocab, trg[0][:trg_len[0]])
                word_error_rates.append(word_error_rate(pred_sents, ref_sents).item())
            except IndexError:
                print(f"The output of the decoder:\n{out_d}\n caused an IndexError!")

            ### print iteration progress ###
            end = time.time()
            if max(1, i) % 50 == 0:
                print("\n" + ("-"*10) + f"Iteration: {i}/{len(dataloader)}" + ("-"*10))
                print(f"Avg loss: {np.mean(losses):.6f}")
                print(f"Avg WER: {np.mean(word_error_rates):.4f}")
                print(f"Time: {(end - start)/60:.4f} min")
                print(f"Predictions: {pred_sents}")
                print(f"References: {ref_sents}")
    
    ### print epoch progross ###
    print("\n" + ("-"*10) + f"VALIDATION" + ("-"*10))
    print(f"Avg WER: {np.mean(word_error_rates)}")
    print(f"Avg loss: {np.mean(losses):.6f}")

    return losses, word_error_rates

def tokenize_targets(target_texts, tokenizer, target_lang_code, max_length, epoch, n_epochs, device):
    padded_targets = tokenizer(text=target_texts, padding=True, return_tensors="pt").input_ids
    return padded_targets.to(device)

def validateMBART(model, dataloaderVal, CFG):
    loss_preds_fc = nn.NLLLoss(
        ignore_index = model.language_model.tokenizer.pad_token_id,
        reduction = "sum").to(CFG.device)
    ctc_loss_fc = torch.nn.CTCLoss(
        blank=0, 
        zero_infinity=True, 
        reduction='sum').to(CFG.device)

    with torch.no_grad():
        model.eval()
        result = compute_metrics(model, dataloaderVal, loss_preds_fc, ctc_loss_fc, tokenize_targets, 0, CFG)
    print("RESULT:\n", result)
    pdb.set_trace()
    #train_pred = model.predict(ipt.to(CFG.device),ipt_len, skip_special_tokens = True)
    #train_for = model.language_model.tokenizer.batch_decode(torch.argmax(preds_permute, dim=1),skip_special_tokens=True)
    #train_target = model.language_model.tokenizer.batch_decode(tokenized_trg_transl, skip_special_tokens=True)
    
class DataPaths:
  def __init__(self):
    self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

VE_CFG = VE_cfg()
S2T_CFG = S2T_cfg()
TRAIN_CFG = Training_cfg()
dp = DataPaths()

### initialize data ###
#train_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
val_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
test_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')
### initialize data ###
#PhoenixTrain = PhoenixDataset(train_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='train')
PhoenixVal = PhoenixDataset(val_df, dp.phoenix_videos, vocab_size=VE_CFG.VOCAB_SIZE, split='dev')
PhoenixTest = PhoenixDataset(test_df, dp.phoenix_videos, vocab_size=VE_CFG.VOCAB_SIZE, split='test')
### get dataloaders ###
#train_augmentations = DataAugmentations(split_type='train')
val_augmentations = DataAugmentations(split_type='val')

#dataloader_train = DataLoader(
#      PhoenixTrain, 
#      collate_fn = lambda data: collator(data, train_augmentations), 
#      batch_size=CFG.batch_size, 
#      shuffle=True, num_workers=CFG.num_workers)

dataloader_val = DataLoader(
    PhoenixVal, 
    collate_fn = lambda data: collator(data, val_augmentations), 
    batch_size=1, 
    shuffle=False,
    num_workers=0)

dataloader_test = DataLoader(
    PhoenixTest, 
    collate_fn = lambda data: collator(data, val_augmentations), 
    batch_size=1, 
    shuffle=False,
    num_workers=0)

criterion = torch.nn.CTCLoss(
        blank=0, 
        zero_infinity=True, 
        reduction='sum').to(VE_CFG.device)

### get decoder and criterion ###
CTC_decoder = ctc_decoder(
    lexicon=None,       
    lm_dict=None,       
    lm=None,            
    tokens= ['-'] + [str(i+1) for i in range(VE_CFG.VOCAB_SIZE)] + ['|'], # vocab + blank and split
    nbest=1, # number of hypotheses to return
    beam_size = 100,       # n.o competing hypotheses at each step
    beam_size_token=50,  # top_n tokens to consider at each step
    beam_threshold = 100) # prune everything below value relative to best score at each step
print("STARTING...")
### initialize model ###
#model = VisualEncoder(CFG).to(CFG.device)

model = Sign2Text(S2T_CFG, VE_CFG).to(VE_CFG.device)

"""
if TRAIN_CFG.load_checkpoint_path is not None:
    from utils.load_checkpoint import load_checkpoint
    import torch.optim as optim
    optimizer = optim.Adam(
        params = model.get_params(TRAIN_CFG), 
        lr=TRAIN_CFG.init_base_lr,
        betas = TRAIN_CFG.betas,
        weight_decay = TRAIN_CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max = TRAIN_CFG.epochs)
    print("\n" + "-"*20 + "Loading Model From Checkpoint" + "-"*20)
    model, optimizer, scheduler, current_epoch, epoch_losses, val_b4 = load_checkpoint(TRAIN_CFG.load_checkpoint_path, model, optimizer, scheduler)
    TRAIN_CFG.start_epoch = current_epoch
"""

### validate Visual Encoder model ###
# validate(model.visual_encoder, dataloader_test, criterion, CTC_decoder, VE_CFG)

### Validate translation model ###
pdb.set_trace()
validateMBART(model, dataloader_val, TRAIN_CFG)