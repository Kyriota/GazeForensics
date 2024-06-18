from enum import Enum


class NormalizationData:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class Normalizations:
    Image_Net = NormalizationData([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    WDF = NormalizationData([0.428, 0.338, 0.301], [0.241, 0.212, 0.215])


class ModelMidSizes(Enum):
    gaze_fc = 0,
    MHA_fc = 1,
    MHA_comp = 2,
    last_fc = 3


class TransformType(Enum):
    standard = 0,
    argument = 1
