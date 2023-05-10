import numpy as np
import evaluate
import torch


def compute_metrics(model, dataloaderTest, loss_preds_fc, ctc_loss_fc, tokenize_targets, CFG):

    metrics = {}
    bleu = evaluate.load("bleu")
    metrics[f"BLEU_1"] = 0
    metrics[f"BLEU_2"] = 0
    metrics[f"BLEU_3"] = 0
    metrics[f"BLEU_4"] = 0
    metrics[f"LOSS"] = 0
    rouge = evaluate.load('rouge')
    metrics["ROUGE"] = 0

    preds = []
    targets = []

    example_index = np.random.randint(len(dataloaderTest))
    for j, datapoint in enumerate(dataloaderTest):
        ipt, ipt_len, trg, trg_len, trg_transl, trg_gloss, max_ipt_len = datapoint

        tokenized_trg_transl = tokenize_targets(
            trg_transl, 
            model.language_model.tokenizer, 
            "de_DE", 
            model.language_model.max_seq_length, 
            CFG.device)

        predicts, probs = model(ipt.to(CFG.device), tokenized_trg_transl, ipt_len)
        preds_permute = predicts.permute(0,2,1)
        probs_permute = probs.permute(1, 0, 2)

        trg = torch.concat([t[:trg_len[i]] for i, t in enumerate(trg)])
        ipt_len = torch.full(size=(probs.size(0),), fill_value = probs.size(1), dtype=torch.int32)
        
        loss = (loss_preds_fc(
            preds_permute, 
            tokenized_trg_transl)
            + 
            ctc_loss_fc(
                torch.log(probs_permute), 
                trg, 
                input_lengths=ipt_len, 
                target_lengths=trg_len))

        metrics[f"LOSS"] += loss.detach().cpu().numpy()

        # raw_preds = model.predict(datapoint[0].to(CFG.device), datapoint[1])
        raw_preds = model.language_model.tokenizer.batch_decode(torch.argmax(preds_permute, dim=1),skip_special_tokens=True)
        raw_targets = datapoint[4]

        for i in range(len(raw_preds)):
            targets.append(raw_targets[i])
            if j == example_index:
                metrics[f"EXAMPLE"] = f"pred: {raw_preds[i]}, target: {raw_targets[i]}"
            if raw_preds[i]:
                preds.append(raw_preds[i])
            else:
                preds.append("@")

    metrics[f"BLEU_1"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 1).get("bleu")
    metrics[f"BLEU_2"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 2).get("bleu")
    metrics[f"BLEU_3"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 3).get("bleu")
    metrics[f"BLEU_4"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 4).get("bleu")
    metrics[f"ROUGE"] += rouge.compute(predictions = preds, references = [[target] for target in targets]).get("rouge1")
    metrics[f"LOSS"] /= len(dataloaderTest)
    
    return metrics