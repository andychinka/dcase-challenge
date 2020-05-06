from torch.utils.data import Dataset
import pandas as pd
import pickle


class Task1aDataSet2018(Dataset):

    def __init__(self, db_path:str, class_map:dict, feature_folder:str, mode:str="train"):
        #mode: train | test | eval

        #read the txt to the data_list
        self.db_path = db_path
        self.class_map = class_map
        df = pd.read_csv("{}/evaluation_setup/fold1_{}.txt".format(db_path, mode), sep="\t", header=None)
        self.X_filepaths = df[0].str.replace("audio", feature_folder, n=1).str.replace(".wav", ".p")

        #TODO: maybe refer to meta.csv
        if mode == "test":
            self.y_classnames = df[0].str.split("/", expand=True)[1].str.split("-", n=1, expand=True)[0]
        else:
            self.y_classnames = df[1]


        self.data_list = []

    def __len__(self):
        return len(self.X_filepaths)

    def __getitem__(self, idx):

        path = self.X_filepaths[idx]
        f = open("{}/{}".format(self.db_path, path), 'rb')
        feature = pickle.load(f)
        label = self.class_map[self.y_classnames[idx]]

        return feature, label