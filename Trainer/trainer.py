import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from utils.get_baseline_metrics import get_baseline_metrics
from utils.compute_metrics import compute_metrics
import time
import numpy as np
import random
from utils.save_checkpoint import save_checkpoint
from utils.load_checkpoint import load_checkpoint
from utils.loss import CustomLoss

def tokenize_targets(target_texts, tokenizer, target_lang_code, max_length, epoch, n_epochs, device):

    # ### probability of masking the input ###
    # mask_prob = epoch / n_epochs
    # tokenizer_mask_token_id = tokenizer.mask_token_id
    
    # ### toknize the target ###
    # tokenized_targets = [[tokenizer.lang_code_to_id["de_DE"]] + tokenizer.encode(
    #     target_text,
    #     add_special_tokens=False,
    # ) + [tokenizer.eos_token_id] for target_text in target_texts]

    # # Pad the tokenized target texts to a maximum length
    # padded_targets = torch.ones((len(target_texts), max_length), dtype=torch.long) * tokenizer.pad_token_id
    # for i, target in enumerate(tokenized_targets):
    #     length = len(target)

    #     ### mask with prob dependent on epoch ###
    #     # for j in range(1,length):
    #     #     if (random.random() < mask_prob):
    #     #         target[j] = tokenizer_mask_token_id

    #     padded_targets[i, :length] = torch.tensor(target[:])
    
    # return padded_targets.to(device)

    padded_targets = tokenizer(text=target_texts, padding=True, return_tensors="pt").input_ids
    return padded_targets.to(device)

def train(model, dataloaderTrain, dataloaderVal, CFG):

    # loss_preds_fc = nn.CrossEntropyLoss(
    #     ignore_index = model.language_model.tokenizer.pad_token_id,
    #     label_smoothing = CFG.ce_label_smoothing).to(CFG.device)
    # loss_preds_fc = nn.NLLLoss(
    #     ignore_index = model.language_model.tokenizer.pad_token_id,
    #     reduction = "sum").to(CFG.device)
    loss_preds_fc = CustomLoss(
        ignore_index = model.language_model.tokenizer.pad_token_id, 
        label_smoothing = CFG.ce_label_smoothing)
    ctc_loss_fc = torch.nn.CTCLoss(
        blank=0, 
        zero_infinity=True, 
        reduction='sum').to(CFG.device)
    optimizer = optim.AdamW(
        params = model.get_params(CFG), 
        lr=CFG.init_base_lr,
        betas = CFG.betas,
        weight_decay = CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max = CFG.epochs)

    if CFG.load_checkpoint_path is not None:
        print("\n" + "-"*20 + "Loading Model From Checkpoint" + "-"*20)
        model, optimizer, scheduler, current_epoch, epoch_losses, val_b4 = load_checkpoint(CFG.load_checkpoint_path, model, optimizer, scheduler)
        CFG.start_epoch = current_epoch
    else:
        epoch_losses = {}

    losses = {}
    epoch_metrics = {}
    epoch_times = {}

    if CFG.verbose:
        print("\n" + "-"*20 + "Preparing Baseline Metrics" + "-"*20)
    
    ###
    baseline_metrics = {}
    # baseline_metrics = get_baseline_metrics(dataloaderVal)
    ###

    if CFG.verbose:
        print("\n" + "-"*20 + "Starting Training" + "-"*20)
    
    for epoch in range(CFG.start_epoch, CFG.epochs):
        losses[epoch] = []
        epoch_start_time = time.time()

        for i, (ipt, ipt_len, trg, trg_len, trg_transl, trg_gloss, max_ipt_len) in enumerate(dataloaderTrain):

            tokenized_trg_transl = tokenize_targets(
                trg_transl, 
                model.language_model.tokenizer, 
                "de_DE", 
                model.language_model.max_seq_length, 
                epoch+1, 
                CFG.epochs,
                CFG.device)

            preds, probs, loss_ce = model(ipt.to(CFG.device), tokenized_trg_transl, ipt_len)
            preds_permute = preds.permute(0,2,1)
            probs_permute = probs.permute(1, 0, 2)

            trg = torch.concat([t[:trg_len[i]] for i, t in enumerate(trg)])
            ipt_len_ctc = torch.full(size=(probs.size(0),), fill_value = probs.size(1), dtype=torch.int32)

            # loss_nll = loss_preds_fc(
            #     nn.functional.log_softmax(preds,dim=-1).contiguous().view(-1,preds.size(-1)), 
            #     tokenized_trg_transl.contiguous().view(-1)) / ipt.size(0)

            loss_nll = loss_preds_fc(
                nn.functional.log_softmax(preds,dim=-1), 
                tokenized_trg_transl) / ipt.size(0)

            loss_ctc = ctc_loss_fc(
                    torch.log(probs_permute), 
                    trg, 
                    input_lengths=ipt_len_ctc, 
                    target_lengths=trg_len) / ipt.size(0)

            loss = loss_nll+loss_ctc

            # loss = (loss_preds_fc(
            #     preds_permute, 
            #     tokenized_trg_transl)
            #     + 
            #     ctc_loss_fc(
            #         torch.log(probs_permute), 
            #         trg, 
            #         input_lengths=ipt_len_ctc, 
            #         target_lengths=trg_len))

            model.zero_grad()
            loss.backward()
            optimizer.step()
            losses[epoch].append(loss.detach().cpu().numpy())

            if CFG.verbose_batches and i % 10 == 0:
                print(f"{i}/{len(dataloaderTrain)}", end="\r", flush=True)
        
        with torch.no_grad():
            model.eval()
            epoch_losses[epoch] = sum(losses[epoch])/len(dataloaderTrain)
            epoch_metrics[epoch] = compute_metrics(model, dataloaderVal, loss_preds_fc, ctc_loss_fc, tokenize_targets, epoch, CFG)
            epoch_times[epoch] = time.time() - epoch_start_time
            train_pred = model.predict(ipt.to(CFG.device),ipt_len, skip_special_tokens = True)
            train_for = model.language_model.tokenizer.batch_decode(torch.argmax(preds_permute, dim=1),skip_special_tokens=True)
            train_target = model.language_model.tokenizer.batch_decode(tokenized_trg_transl, skip_special_tokens=True)
            model.train()
        
        if CFG.verbose:
            print("\n" + "-"*50)
            print(f"EPOCH: {epoch}")
            print(f"TIME: {epoch_times[epoch]}")
            print(f"AVG. LOSS: {epoch_losses[epoch]}")
            print(f"EPOCH METRICS: {epoch_metrics[epoch]}")
            print(f"BASELINE METRICS: {baseline_metrics}")
            print(train_pred)
            print(train_for)
            print(train_target)
            print("-"*50)

        scheduler.step()

        ### save model ### 
        if CFG.save_checkpoints and epoch % 2 == 0:
            save_path = CFG.save_path +  "Sign2Text_Epoch" + str(epoch+1) + "_loss_" + str(epoch_losses[epoch]) +  "_B4_" + str(epoch_metrics[epoch]["BLEU_4"])
            save_checkpoint(save_path, model, optimizer, scheduler, epoch, epoch_losses, epoch_metrics[epoch]["BLEU_4"])

    return losses, epoch_losses, epoch_metrics, epoch_times