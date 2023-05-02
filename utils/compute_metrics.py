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

    preds = []
    targets = []

    example_index = np.random.randint(len(dataloaderTest))
    for j, datapoint in enumerate(dataloaderTest):
        raw_preds = model.predict(datapoint[0].to(CFG.device), datapoint[1])
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
    
    return metrics