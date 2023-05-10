
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
        self.epochs = 100
        self.start_epoch = 0
        self.batch_size = 4
        self.num_workers = 8
        ### verbose ###
        self.verbose = True
        self.verbose_batches = False
        ### paths ###
        self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
        self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
        ### saving models ###
        self.save_path = '/work3/s200925/Sign2Text/new_models/'
        self.save_checkpoints = False
        ### loading model ###
        self.load_checkpoint_path = '/work3/s200925/Sign2Text/new_models/Sign2Text_Epoch25_loss_23.816143760148943_B4_0.011217397415332899' # None # '/work3/s200925/Sign2Text/new_models/Sign2Text_Epoch31_loss_23.629026475861217_B4_0.0'
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('mps')