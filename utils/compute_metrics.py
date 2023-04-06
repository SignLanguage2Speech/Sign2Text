import numpy as np
import evaluate


def compute_metrics(model, dataloaderTest, CFG):

    metrics = {}
    bleu = evaluate.load("bleu")
    metrics[f"BLEU_1"] = 0
    metrics[f"BLEU_2"] = 0
    metrics[f"BLEU_3"] = 0
    metrics[f"BLEU_4"] = 0
    rouge = evaluate.load('rouge')
    metrics["ROUGE"] = 0

    hasExample = False
    for datapoint in dataloaderTest:
        raw_preds = model.predict(datapoint[0].to(CFG.device))
        raw_targets = datapoint[4]

        preds = []
        targets = []
        for i in range(len(raw_preds)):
            if not hasExample:
                metrics[f"EXAMPLE"] = f"pred: {raw_preds[i]}, target: {raw_targets[i]}"
                hasExample = True
            if raw_preds[i]:
                preds.append(raw_preds[i])
                targets.append(raw_targets[i])

        if len(preds) > 0:
            metrics[f"BLEU_1"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 1).get("bleu")
            metrics[f"BLEU_2"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 2).get("bleu")
            metrics[f"BLEU_3"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 3).get("bleu")
            metrics[f"BLEU_4"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 4).get("bleu")
            metrics[f"ROUGE"] += rouge.compute(predictions = preds, references = [[target] for target in targets]).get("rouge1")

    for metric in metrics.keys():
        if not metric == "EXAMPLE":
            metrics[metric] /= len(dataloaderTest)
    
    return metrics