from mld.models.motionclip_263.models.architectures.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER
from mld.models.motionclip_263.models.modeltype.motionclip import MOTIONCLIP

JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

LOSSES = ["rc", "rcxyz", "vel", "velxyz"]  # not used: "hp", "mmd", "vel", "velxyz"

def get_model(parameters, clip_model):
    encoder = Encoder_TRANSFORMER(**parameters)
    decoder = Decoder_TRANSFORMER(**parameters)
    parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
    return MOTIONCLIP(encoder, decoder, clip_model=clip_model, **parameters).to(parameters["device"])
