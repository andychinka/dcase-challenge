import openl3
import soundfile as sf
import numpy as np
import os
import argparse

from asc.preprocess.base import PreProcessBase

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class OpenL3PreProcess(PreProcessBase):

    def __init__(self, db_path: str, feature_folder: str,
                 content_type: str, input_repr: str, embedding_size: int, hop_size: float):
        super().__init__(db_path, feature_folder)
        self.hop_size = hop_size

        self.model = openl3.models.load_audio_embedding_model(content_type=content_type,
                                                              input_repr=input_repr,
                                                              embedding_size=embedding_size)

    def extract_feature(self, wave_fp: str):

        x, sr = sf.read(wave_fp)

        #TODO: check x.shape
        # assert (x.shape == (441000,))
        if x.shape[0] == sr*10 - 1:
            x = np.append(x, 0)
        elif x.shape[0] == sr*10 + 1:
            x = x[:-1]
        assert (x.shape == (sr*10,)) # suppose audio are in 10s

        emb, ts = openl3.get_audio_embedding(x, sr, model=self.model, hop_size=self.hop_size)

        return np.expand_dims(emb, axis=0)  # (1, xx, embedding_size), xx=96 when hop_size=0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-db_path', default="/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development")
    parser.add_argument("-feature_folder", default="openl3-music-mel256-emb512-hop0_1")

    parser.add_argument("-content_type", default="music")
    parser.add_argument("-input_repr", default="mel256")
    parser.add_argument("-embedding_size", default=512)
    parser.add_argument("-hop_size", default=0.1)

    args = parser.parse_args()

    preProcess = OpenL3PreProcess(args.db_path, args.feature_folder,
                                  args.content_type, args.input_repr, args.embedding_size, args.hop_size)
    preProcess.process()
    print("done.")