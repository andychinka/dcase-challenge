import librosa
import numpy as np
import soundfile as sound

from asc.preprocess.base import PreProcessBase


# ref: https://github.com/McDonnell-Lab/DCASE2019-Task1
class LogMelHTKPreProcess(PreProcessBase):

    def __init__(self, db_path: str, feature_folder: str,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = int(2048/2),
                 sample_rate: int = 44100,
                 deltas_deltas: bool = True):
        super().__init__(db_path, feature_folder)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.deltas_deltas = deltas_deltas

    def extract_feature(self, wave_fp):
        stereo, fs = sound.read(wave_fp, frames=10 * self.sample_rate, fill_value=0) #(441000) #stop=10 * self.sample_rate,
        if len(stereo.shape)==1:
            stereo = np.expand_dims(stereo, 0) #(1, 441000)

        feature = np.zeros((stereo.shape[0], self.n_mels, int(np.ceil(10 * self.sample_rate/self.hop_length))), 'float32')
        for channel in range(stereo.shape[0]):
            feature[channel, :, :] = librosa.feature.melspectrogram(stereo[channel, :],
                                                     sr=self.sample_rate,
                                                     n_fft=self.n_fft,
                                                     hop_length=self.hop_length,
                                                     n_mels=self.n_mels,
                                                     fmin=0.0,
                                                     fmax=self.sample_rate / 2,
                                                     htk=True,
                                                     norm=None) #(1, 128, 431)
            feature = np.log(feature + 1e-8)

            if self.deltas_deltas:
                feature_deltas = self.deltas(feature) #(1, 128, 427)
                feature_delats_delats = self.deltas(feature_deltas) #(1, 128, 423)
                feature = np.concatenate((
                    feature[:, :, 4:-4],
                    feature_deltas[:, :, 2:-2],
                    feature_delats_delats
                ), axis=0) #(3, 128, 423)

        return feature

    def deltas(self, X_in):
        X_out = (X_in[:,:,2:]-X_in[:,:,:-2])/10.0
        X_out = X_out[:,:,1:-1]+(X_in[:,:,4:]-X_in[:,:,:-4])/5.0
        return X_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-db_path', default="/home/MSAI/ch0001ka/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development")
    parser.add_argument("-feature_folder", default="logmel_128_44k_htk")

    parser.add_argument("-n_mels", default=128)
    parser.add_argument("-n_fft", default=2048)
    parser.add_argument("-hop_length", default=int(2048/2))
    parser.add_argument("-sample_rate", default=44100)
    parser.add_argument("-deltas_deltas", default=False)

    args = parser.parse_args()

    preProcess = LogMelHTKPreProcess(args.db_path, args.feature_folder,
                                  args.n_mels, args.n_fft, args.hop_length,
                                  args.sample_rate, args.deltas_deltas)
    preProcess.process()
    print("done.")