from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from asc.preprocess.log_mel_htk_preprocess import LogMelHTKPreProcess

class Task1bDataSet2019Test(Dataset):

    def __init__(self, db_path:str):
        #mode: train | test | eval
        #device：a | b | c

        #read the txt to the data_list
        self.db_path = db_path
        df = pd.read_csv("{}/evaluation_setup/test.csv".format(db_path), sep="\t")
        self.X_filepaths = df["filename"] #.str.replace("audio", feature_folder, n=1).str.replace(".wav", ".npy")
        self.data_list = []
        self.preProcess = LogMelHTKPreProcess(db_path, "",
                                  128, 2048, int(2048/2),
                                  44100, True)

    def __len__(self):
        return len(self.X_filepaths)

    def __getitem__(self, idx):

        path = self.X_filepaths[idx]
        feature = self.preProcess.extract_feature("{}/{}".format(self.db_path, path))
        # f = open("{}/{}".format(self.db_path, path), 'rb')
        # feature = np.load(f)

        return feature

