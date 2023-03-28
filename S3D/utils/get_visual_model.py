from S3D.utils.load_weights import load_PHOENIX_weights
from S3D.s3d_backbone import VisualEncoder

def get_visual_model(visual_model_vocab_size, visual_model_checkpoint, device):
        visual_model = VisualEncoder(visual_model_vocab_size + 1).to(device)
        load_PHOENIX_weights(visual_model, visual_model_checkpoint, verbose=False)
        for param in visual_model.state_dict():
                if "base" in param:
                        visual_model.state_dict()[param].requires_grad = False
        return visual_model