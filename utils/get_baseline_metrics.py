import numpy as np
import evaluate


def get_baseline_metrics(dataloaderTest):
    
    metrics = {}
    bleu = evaluate.load("bleu")
    metrics[f"BLEU_1"] = 0
    metrics[f"BLEU_2"] = 0
    metrics[f"BLEU_3"] = 0
    metrics[f"BLEU_4"] = 0
    rouge = evaluate.load('rouge')
    metrics["ROUGE"] = 0

    for batch in dataloaderTest:
        preds = batch[2]
        targets = batch[4]

        metrics[f"BLEU_1"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 1).get("bleu")
        metrics[f"BLEU_2"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 2).get("bleu")
        metrics[f"BLEU_3"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 3).get("bleu")
        metrics[f"BLEU_4"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 4).get("bleu")
        metrics[f"ROUGE"] += rouge.compute(predictions = preds, references = [[target] for target in targets]).get("rouge1")

    for metric in metrics.keys():
        metrics[metric] /= len(dataloaderTest)
    
    return metrics