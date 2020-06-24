import numpy as np
import os


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

            f_fp = wav_fp.replace(".wav", ".npy", 1)
            trg_fp = "{}/{}/{}".format(self.db_path, self.feature_folder, f_fp)
            if os.path.exists(trg_fp) and os.path.getsize(trg_fp) > 0:
                print("already extracted, skip, ", trg_fp)
                continue

            f = self.extract_feature("{}/audio/{}".format(self.db_path, wav_fp))
            with open(trg_fp, 'wb') as f_file:
                np.save(f_file, f)
                # pickle.dump(f, f_file)