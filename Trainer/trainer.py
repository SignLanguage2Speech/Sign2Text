import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from utils.get_baseline_metrics import get_baseline_metrics
from utils.compute_metrics import compute_metrics
import time
import numpy as np

def tokenize_targets(target_texts, tokenizer, target_lang_code, max_length, device):
    tokenized_targets = [tokenizer.encode(
        target_text,
        add_special_tokens=False,
    ) + [tokenizer.eos_token_id] for target_text in target_texts]

    # Add the [lang_code] token at the beginning of each target text
    tokenized_targets = [[tokenizer.lang_code_to_id[target_lang_code]] + target_text for target_text in tokenized_targets]

    # Pad the tokenized target texts to a maximum length
    padded_targets = torch.ones((len(target_texts), max_length), dtype=torch.long) * tokenizer.pad_token_id
    for i, target in enumerate(tokenized_targets):
        length = min(len(target), max_length)
        padded_targets[i, :length] = torch.tensor(target[:length])
    
    return padded_targets.to(device)

def train(model, dataloaderTrain, dataloaderVal, CFG):

    loss_preds_fc = nn.CrossEntropyLoss(
        # ignore_index = 1,
        label_smoothing=CFG.ce_label_smoothing)
    ctc_loss_fc = nn.CTCLoss()
    optimizer = optim.Adam(
        params = model.get_params(CFG), 
        lr=CFG.init_base_lr,
        betas = CFG.betas,
        weight_decay = CFG.weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max = CFG.epochs)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1,
        end_factor = 0.01,
        total_iters = CFG.epochs)


    losses = {}
    epoch_losses = {}
    epoch_metrics = {}
    epoch_times = {}

    if CFG.verbose:
        print("\n" + "-"*20 + "Preparing Baseline Metrics" + "-"*20)
    
    baseline_metrics = get_baseline_metrics(dataloaderVal)

    if CFG.verbose:
        print("\n" + "-"*20 + "Starting Training" + "-"*20)
    
    for epoch in range(CFG.epochs):
        losses[epoch] = []
        epoch_start_time = time.time()

        for i, (ipt, ipt_len, trg, trg_len, trg_transl, trg_gloss, max_ipt_len) in enumerate(dataloaderTrain):

            tokenized_trg_transl = tokenize_targets(
                trg_transl, 
                model.language_model.tokenizer, 
                "de_DE", 
                int(np.ceil(max_ipt_len/4)), 
                CFG.device)

            preds, probs = model(ipt.to(CFG.device))
            preds_permute = preds.permute(0,2,1)
            probs_permute = probs.permute(1, 0, 2)

            trg = torch.concat([t[:trg_len[i]] for i, t in enumerate(trg)])
            ipt_len = torch.full(size=(probs.size(0),), fill_value = probs.size(1), dtype=torch.int32)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss = (loss_preds_fc(
                    preds_permute, 
                    tokenized_trg_transl)
                    + 
                    ctc_loss_fc(
                        torch.log(probs_permute), 
                        trg, 
                        input_lengths=ipt_len, 
                        target_lengths=trg_len))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            losses[epoch].append(loss.detach().cpu().numpy())

            if CFG.verbose_batches and i % 500 == 0:
                print(f"{i}/{len(dataloaderTrain)}", end="\r", flush=True)
        
        with torch.no_grad():
            model.eval()
            epoch_losses[epoch] = sum(losses[epoch])/len(dataloaderTrain)
            epoch_metrics[epoch] = compute_metrics(model, dataloaderVal, CFG)
            epoch_times[epoch] = time.time() - epoch_start_time
            model.train()
        
        if CFG.verbose:
            print("\n" + "-"*50)
            print(f"EPOCH: {epoch}")
            print(f"TIME: {epoch_times[epoch]}")
            print(f"AVG. LOSS: {epoch_losses[epoch]}")
            print(f"EPOCH METRICS: {epoch_metrics[epoch]}")
            print(f"BASELINE METRICS: {baseline_metrics}")
            print(model.predict(ipt.to(CFG.device)))
            print(trg_transl)
            print("-"*50)

        scheduler.step()

    return losses, epoch_losses, epoch_metrics, epoch_times