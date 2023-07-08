from .vesselformer import build_vesselformer

def build_model(config, **kwargs):
    return build_vesselformer(config, **kwargs)