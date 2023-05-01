
import torch

class Training_cfg:
    def __init__(self):
        ### training params ###
        self.ce_label_smoothing = 0.2
        self.init_base_lr = 0.001
        self.init_lr_language_model = 0.0001
        self.init_lr_visual_model = 0.001
        self.betas = (0.9, 0.998)
        self.weight_decay = 1.0e-3
        self.epochs = 250
        self.batch_size = 2
        self.num_workers = 0
        ### verbose ###
        self.verbose = True
        self.verbose_batches = False
        ### paths ###
        self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
        self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')