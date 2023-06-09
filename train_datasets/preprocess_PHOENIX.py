import os
import pandas as pd
import re
from itertools import groupby

def preprocess_df(df, split, save=False, save_name = "PHOENIX_train_preprocessed.csv", use_synthetic_glosses=False):
    annotations_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
    features_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    gloss_vocab, translation_vocab = getVocab(annotations_path, use_synthetic_glosses)

    print("-"*10 + f"GLOSS VOCAB SIZE {len(gloss_vocab)}" + "-" * 10)

    # add translation and gloss labels
    df = getLabels(df, translation_vocab, gloss_vocab, use_synthetic_glosses)
    if save:
        df.to_csv(os.path.join(annotations_path, save_name))
    
    return df

def getLabels(df, t_vocab, g_vocab, use_synthetic_glosses):

    all_translations = []
    all_glosses = []

    g_vocab_size = len(g_vocab.keys())
    t_vocab_size = len(t_vocab.keys())
    for i in range(len(df)):
        T = df.iloc[i]['translation'].split(' ')
        if not use_synthetic_glosses:
            G = clean_phoenix_glosses(df.iloc[i]['orth']).split(' ')
        else:
            G = df.iloc[i]['synthetic glosses'].split(' ')
        T_labels = []
        G_labels = []
        
        for word in T:
            try:
                T_labels.append(t_vocab[word])
            except KeyError:
                T_labels.append(t_vocab_size + 1) # Handle OOV in validation & test
        
        for gloss in G:
            try:
                G_labels.append(g_vocab_size + 1)
            except KeyError:
                G_labels.append(1086) # Handle OOV in validation & test
                
        all_translations.append(T_labels)
        all_glosses.append(G_labels)
    df['translation_labels'] = all_translations
    df['gloss_labels'] = all_glosses

    return df

def getVocab(path, use_synthetic_glosses=False):
    if not use_synthetic_glosses:
        train = pd.read_csv(os.path.join(path, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
        glosses = list(clean_phoenix_glosses(train.iloc[i]['orth']) for i in range(len(train)))
    else:
        train = pd.read_csv(os.path.join(path, 'PHOENIX-2014-T.train.corpus.synthetic.glosses.csv'))
        glosses = list([train.iloc[i]['synthetic glosses'] for i in range(len(train))])
    
    chars = '?.,!_+'
    # get vocabulary for translations and glosses for the train dataset
    glosses = list(sorted(set([word for sent in glosses for word in sent.replace(chars,'').split(' ')])))

    translations = list(train['translation'])
    translations = list(sorted(set([word for sent in translations for word in sent.replace(chars,'').split(' ')])))
    
    gloss_vocab = {word: glosses.index(word)+1 for word in glosses}
    translation_vocab = {word: translations.index(word)+1 for word in translations}
    return gloss_vocab, translation_vocab


def clean_phoenix_glosses(prediction):

    prediction = prediction.strip()
    prediction = re.sub(r"__LEFTHAND__", "", prediction)
    prediction = re.sub(r"__EPENTHESIS__", "", prediction)
    prediction = re.sub(r"__EMOTION__", "", prediction)
    prediction = re.sub(r"\b__[^_ ]*__\b", "", prediction)
    prediction = re.sub(r"\bloc-([^ ]*)\b", r"\1", prediction)
    prediction = re.sub(r"\bcl-([^ ]*)\b", r"\1", prediction)
    prediction = re.sub(r"\b([^ ]*)-PLUSPLUS\b", r"\1", prediction)
    prediction = re.sub(r"\b([A-Z][A-Z]*)RAUM\b", r"\1", prediction)
    prediction = re.sub(r"WIE AUSSEHEN", "WIE-AUSSEHEN", prediction)
    prediction = re.sub(r"^([A-Z]) ([A-Z][+ ])", r"\1+\2", 
            prediction)
    prediction = re.sub(r"[ +]([A-Z]) ([A-Z]) ", r" \1+\2 ", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +]SCH) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +]NN) ([A-Z][ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) (NN[ +])", r"\1+\2", prediction)
    prediction = re.sub(r"([ +][A-Z]) ([A-Z])$", r"\1+\2", prediction)
    prediction = re.sub(r" +", " ", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", prediction)
    prediction = re.sub(r" +", " ", prediction)

    prediction = " ".join(
        " ".join(i[0] for i in groupby(prediction.split(" "))).split()
    )
    prediction = prediction.strip()

    return prediction

"""
### Deprecated! ###

def addLengths(df, features_path, cutoff=330):
  video_names = list(df['name'])
  lengths = []
  for name in video_names:
    length = len(os.listdir(os.path.join(features_path, 'train', name)))
    lengths.append(length)
  
  df['video_length'] = lengths
  df = df[df['video_length'] < cutoff]
  return df.reset_index()

def groupByBin(df):
  ### calculate bins
  min_val = min(df['video_length'])
  max_val = max(df['video_length'])
  bins = [min_val]
  val = min_val

  while val < max_val:
    val = int(val*1.25)
    bins.append(val)

  dataframes = []
  for i in range(len(bins)-1):
    df_new = df[(df['video_length'] >= bins[i]) & (df['video_length'] < bins[i+1])]
    dataframes.append(df_new)

  return dataframes
"""
