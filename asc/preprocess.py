import pandas as pd
import pickle
import librosa
import numpy as np
import os
import argparse

class PreProcessBase(object):

    def __init__(self, db_path: str, feature_folder: str = "feature"):
        self.db_path = db_path
        self.feature_folder = feature_folder

        if not os.path.exists("{}/{}".format(db_path, feature_folder)):
            os.mkdir("{}/{}".format(db_path, feature_folder))

    def extract_feature(self, wave_fp: str):
        raise Exception("Please implement this function")

    def process(self):

        for wav_fp in os.listdir("{}/audio".format(self.db_path)):
            if not wav_fp.endswith(".wav"):
                continue
            f = self.extract_feature("{}/audio/{}".format(self.db_path, wav_fp))

            f_fp = wav_fp.replace(".wav", ".p", 1)

            with open("{}/{}/{}".format(self.db_path, self.feature_folder, f_fp), 'wb') as f_file:
                pickle.dump(f, f_file)


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
    parser.add_argument('-db_path', default="/media/data/shared-data/dcase/TUT-urban-acoustic-scenes-2018-development")
    parser.add_argument("-feature_folder", default="feature2")

    parser.add_argument("-n_fft", default=2048)
    parser.add_argument("-hop_length_sec", default=0.02)
    parser.add_argument("-win_length_sec", default=0.04)
    parser.add_argument("-n_mels", default=40)
    parser.add_argument("-fmax", default=24000)

    args = parser.parse_args()

    preProcess = LogMelPreProcess(args.db_path, args.feature_folder,
                                  args.n_fft, args.hop_length_sec, args.win_length_sec, args.n_mels, args.fmax)
    preProcess.process()
    print("done.")
