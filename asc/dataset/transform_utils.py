import librosa
import numpy as np


class SelectChannel(object):

    def __init__(self, select_index):
        self.select_index = select_index

    def __call__(self, feature):

        feature = feature[self.select_index]

        return np.expand_dims(feature, axis=0)


class Normalizer(object):

    def __call__(self, feature):

        return librosa.util.normalize(feature, axis=1)