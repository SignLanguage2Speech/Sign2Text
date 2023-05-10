from train_datasets.preprocess_PHOENIX import getVocab
import torch

class cfg:
    def __init__(self) -> None:
        self.n_classes = 1085 + 1 # +1 for blank token
        self.VOCAB_SIZE = self.n_classes - 1
        # S3D backbone
        self.use_block = 4 # use everything except lass block
        self.freeze_block = 4 # 4! [0, ...5] 
        # Head network
        self.ff_size = 2048
        self.input_size = 832
        self.hidden_size = 512
        self.ff_kernel_size = 3
        self.residual_connection = True
        self.head_dropout = 0.10 # 0.10 in SOTA config
        # verbose for weightloading #
        self.verbose = False
        ### paths ###
        self.backbone_weights_filename = None #'/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc'
        self.head_weights_filename = None
        self.checkpoint_path = '/work3/s204138/bach-models/Kinetics_CTC_training/S3D_PHOENIX-100_epochs-12.429813_loss_0.229911_WER' # None  # if None train from scratch
        self.gloss_vocab, self.translation_vocab = getVocab('/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual')
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

