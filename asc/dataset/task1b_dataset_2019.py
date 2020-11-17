from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class Task1bDataSet2019(Dataset):

    def __init__(self, db_path:str, class_map:dict, feature_folder:str, mode:str="train", device: str = None, transform=None):
        #mode: train | test | eval
        #device：a | b | c

        #read the txt to the data_list
        self.db_path = db_path
        self.class_map = class_map
        df = pd.read_csv("{}/evaluation_setup/fold1_{}.csv".format(db_path, mode), sep="\t")
        self.X_filepaths = df["filename"].str.replace("audio", feature_folder, n=1).str.replace(".wav", ".npy")
        self.transform = transform

        #TODO: maybe refer to meta.csv
        if mode == "test":
            if device is None:
                self.y_classnames = df["filename"].str.split("/", expand=True)[1].str.split("-", n=1, expand=True)[0]
            else:
                fp_list = []
                y_classnames = []
                trg_suffix = "-{}.npy".format(device)
                for index, fp in self.X_filepaths.items():
                    if not fp[-4:] == trg_suffix:
                        continue
                    fp_list.append(fp)
                    y_classnames.append(fp.split("/")[1].split("-", 1)[0])

                self.X_filepaths = fp_list
                self.y_classnames = y_classnames

        else:
            self.y_classnames = df["scene_label"]


        self.data_list = []

    def __len__(self):
        return len(self.X_filepaths)

    def __getitem__(self, idx):

        path = self.X_filepaths[idx]

        scene, city, device = self.parse_filename(path)

        f = open("{}/{}".format(self.db_path, path), 'rb')
        feature = np.load(f)
        label = self.class_map[self.y_classnames[idx]]

        if self.transform:
            feature = self.transform(feature)

        return feature, label, city, device

    def parse_filename(self, path: str):
        # Path: <feature_folder>/<scene>-<city>-<###>-<###>-<device>.xxx
        path = path.split(".")[0]  # remove the file extension part
        filename = path.split("/")[-1] # remove feature folder
        filename_parts = filename.split("-")

        scene, city, device = None, None, None

        if len(filename_parts) == 5:
            scene = filename_parts[0]
            city = filename_parts[1]
            device = filename_parts[-1]

        return scene, city, device
