import torch

class Sign2Text_cfg:
    def __init__(self):
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ### model params ###
        self.mbart_path = '/work3/s200925/mBART/checkpoints/checkpoint-70960'
        self.beam_width = 4
        self.max_seq_length = 100
        self.length_penalty = 1
        self.mbart_dropout = 0.3
        self.mbart_attention_dropout = 0.1
        ### ??? ###
        self.n_visual_features = 512
        self.n_classes = 1085 + 1

