
import torch

class Training_cfg:
    def __init__(self):
        ### training params ###
        self.ce_label_smoothing = 0.2
        self.init_base_lr = 0.001
        self.init_lr_language_model = 0.00001
        self.init_lr_visual_model = 0.001
        self.betas = (0.9, 0.998)
        self.weight_decay = 1.0e-3
        self.epochs = 80
        self.start_epoch = 0
        self.batch_size = 4
        self.num_workers = 8
        ### verbose ###
        self.verbose = True
        self.verbose_batches = True
        ### paths ###
        self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
        self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
        ### saving models ###
        self.save_path = '/work3/s200925/Sign2Text/new_loss/'
        self.save_checkpoints = True
        ### loading model ###
        self.load_checkpoint_path =  None # '/work3/s200925/Sign2Text/no_mask_dropout_03/Sign2Text_Epoch28_loss_21.39952958127423_B4_0.10217316517638868'
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')