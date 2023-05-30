import torch

class Sign2Text_cfg:
    def __init__(self):
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('mps')
        ### model params ###
        self.mbart_path = 'model/checkpoint-70960'
        self.beam_width = 4
        self.max_seq_length = 100
        self.length_penalty = 1
        self.VL_mapper_dropout = 0.0
        ### ??? ###
        self.n_visual_features = 512
        self.n_classes = 1085 + 1
        ### cc25 params
        self.vocab_data = 'mBART/german_data.txt'

