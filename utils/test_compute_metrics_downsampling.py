import numpy as np
import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_compute_metrics_downsampling(model, dataloaderTest, downsampling_rate, CFG):

    metrics = {}
    bleu = evaluate.load("bleu")
    metrics[f"BLEU_1"] = 0
    metrics[f"BLEU_2"] = 0
    metrics[f"BLEU_3"] = 0
    metrics[f"BLEU_4"] = 0
    rouge = evaluate.load('rouge')
    metrics["ROUGE"] = 0

    preds = []
    targets = []

    example_index = np.random.randint(len(dataloaderTest))
    for j, datapoint in enumerate(dataloaderTest):
        ipt, ipt_len, trg, trg_len, trg_transl, trg_gloss, max_ipt_len = datapoint
        
        if not downsampling_rate == 1:
            ipt = temporal_downsample(ipt[0], downsampling_rate).unsqueeze(0)
            ipt_len[0] = ipt.size()[2]

        raw_preds = model.predict(ipt.to(CFG.device), ipt_len)
        raw_targets = trg_transl

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
    
    return metrics

def temporal_downsample(rawvideo, scale_factor):
        rawvideo = rawvideo.permute(0,2,3,1)
        mode = 'nearest'
        #return F.interpolate(rawvideo, scale_factor=self.k_t, mode=mode)#, align_corners=False, recompute_scale_factor=None)
        c0 = F.interpolate(rawvideo[0], scale_factor=scale_factor, mode=mode)
        c1 = F.interpolate(rawvideo[1], scale_factor=scale_factor, mode=mode)
        c2 = F.interpolate(rawvideo[2], scale_factor=scale_factor, mode=mode)
        return torch.stack([c0,c1,c2]).permute(0,3,1,2)