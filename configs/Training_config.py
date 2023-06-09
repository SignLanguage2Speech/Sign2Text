
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
        self.epochs = 40
        self.start_epoch = 0
        self.batch_size = 6
        self.num_workers = 8
        ### verbose ###
        self.verbose = True
        self.verbose_batches = True
        ### paths ###
        self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
        self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
        ### saving models ###
        self.save_path = '/work3/s200925/Sign2Text/synthetic_training/'
        self.save_checkpoints = True
        ### loading model ###
        self.load_checkpoint_path = '/work3/s200925/Sign2Text/synthetic_training/Sign2Text_Epoch9_loss_-35.31798725499002_B4_0.14294142376260746' #'/work3/s200925/Sign2Text/final_training/Sign2Text_Epoch49_loss_16.049303379687633_B4_0.2520139900900941' # '/work3/s200925/Sign2Text/label_smooth/Sign2Text_Epoch37_loss_16.43221887015436_B4_0.2297965381199748'
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ### ablation study ###
        self.synthetic_glosses=False