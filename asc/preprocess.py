import pandas as pd
import pickle
import librosa
import numpy as np
import os
import argparse
from asc.preprocess.base import PreProcessBase


class LogMelPreProcess(PreProcessBase):

    def __init__(self, db_path: str, feature_folder: str,
                 n_fft, hop_length_sec, win_length_sec, n_mels, fmax):
        super().__init__(db_path, feature_folder)
        self.n_fft = n_fft
        self.hop_length_sec = hop_length_sec
        self.win_length_sec = win_length_sec
        self.n_mels = n_mels
        self.fmax = fmax

    def extract_feature(self, wave_fp: str):
        x, sr = librosa.load(wave_fp, sr=None, mono=True)
        # assert (x.shape == (441000,))
        if x.shape[0] == 479999 or x.shape[0] == 440999:
            x = np.append(x, 0)
        elif x.shape[0] == 480001 or x.shape[0] == 441001:
            x = x[:-1]
        assert (x.shape == (sr*10,))
        # 40ms winlen, half overlap
        y = librosa.feature.melspectrogram(x,
                                           sr=sr,
                                           n_fft=self.n_fft,
                                            hop_length=int(sr*self.hop_length_sec),
                                            win_length=int(sr*self.win_length_sec),
                                           n_mels=self.n_mels,
                                           fmax=self.fmax
                                           )
        # about 1e-7
        EPS = np.finfo(np.float32).eps
        fea = np.log(y + EPS)
        # add a new axis
        return np.expand_dims(fea[:, :-1], axis=0)  # (1, 40, 500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-db_path', default="/media/data/shared-data/dcase/TAU-urban-acoustic-scenes-2019-development")
    parser.add_argument("-feature_folder", default="feature2")

    parser.add_argument("-n_fft", default=2048)
    parser.add_argument("-hop_length_sec", default=0.02)
    parser.add_argument("-win_length_sec", default=0.04)
    parser.add_argument("-n_mels", default=40)
    parser.add_argument("-fmax", default=24000) # task1A: 24000, task1B: 22050

    args = parser.parse_args()

    preProcess = LogMelPreProcess(args.db_path, args.feature_folder,
                                  args.n_fft, args.hop_length_sec, args.win_length_sec, args.n_mels, args.fmax)
    preProcess.process()
    print("done.")
