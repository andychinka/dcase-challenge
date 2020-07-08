import librosa
import numpy as np

from asc.preprocess.base import PreProcessBase


class LogMelPreProcess(PreProcessBase):

    def __init__(self, db_path: str, feature_folder: str,
                 n_mels: int = 40,
                 n_fft: int = int(0.04*48000),
                 hop_length: int = int(0.02*48000),
                 win_length: int = int(0.04*48000),
                 sample_rate: int = 48000,
                 deltas: bool = False):
        super().__init__(db_path, feature_folder)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.deltas = deltas

    def extract_feature(self, wave_fp):
        x, sr = librosa.load(wave_fp, self.sample_rate)

        feature = librosa.feature.melspectrogram(x,
                            sr=self.sample_rate,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            n_mels=self.n_mels)

        feature = np.log10(np.maximum(feature, 1e-10))

        if (feature.shape[-1] - 500) < 3 and (feature.shape[-1] - 500) > 0:
            feature = feature[:,:500]

        assert(feature.shape[-1] == 500)
        feature = np.expand_dims(np.flip(feature), axis=0) # (1, 40, 500)

        #TODO: handling for deltas
        # if self.deltas:
        #     F = logmeldeltasx3(F)

        return feature


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-db_path', default="/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development")
    parser.add_argument("-feature_folder", default="logmel_40_44k")

    parser.add_argument("-n_mels", default=40)
    parser.add_argument("-n_fft", default=int(0.04 * 44100))
    parser.add_argument("-hop_length", default=int(0.02 * 44100))
    parser.add_argument("-win_length", default=int(0.04 * 44100))
    parser.add_argument("-sample_rate", default=44100)
    parser.add_argument("-deltas", default=False)

    args = parser.parse_args()

    preProcess = LogMelPreProcess(args.db_path, args.feature_folder,
                                  args.n_mels, args.n_fft, args.hop_length, args.win_length,
                                  args.sample_rate, args.deltas)
    preProcess.process()
    print("done.")