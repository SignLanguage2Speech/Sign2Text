from utils import sacrebleu

def bleu(references, hypotheses, level='word'):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)
    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    if level=='char':
        #split word
        references = [' '.join(list(r)) for r in references]
        hypotheses = [' '.join(list(r)) for r in hypotheses]
    bleu_scores = sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores