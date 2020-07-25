from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
# from specAugment import spec_augment_tensorflow

class Task1aDataSet2020(Dataset):

    def __init__(self, db_path:str, class_map:dict, feature_folder:str, mode:str="train", device: str = None):
        #mode: train | test | eval
        #device：a | b | c

        #read the txt to the data_list
        self.db_path = db_path
        self.class_map = class_map
        df = pd.read_csv("{}/evaluation_setup/fold1_{}.csv".format(db_path, mode), sep="\t")
        self.X_filepaths = df["filename"].str.replace("audio", feature_folder, n=1).str.replace(".wav", ".npy")

        #TODO: maybe refer to meta.csv
        if mode == "test":
            if device is None:
                self.y_classnames = df["filename"].str.split("/", expand=True)[1].str.split("-", n=1, expand=True)[0]
            else:
                fp_list = []
                y_classnames = []
                trg_suffix = "-{}.npy".format(device)
                trg_suffix_len = len(trg_suffix)
                for index, fp in self.X_filepaths.items():
                    if not fp[-trg_suffix_len:] == trg_suffix:
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
        f = open("{}/{}".format(self.db_path, path), 'rb')
        feature = np.load(f)

        #### no need do swap and squeeze for cnn, code chnaged to accept (1, dim, time)
        # if feature.shape[0] == 1:
        #     feature = np.squeeze(feature, axis=0) # for dim256, cnn9
        # feature = np.swapaxes(feature, 0, 1) # only for cnn

        # feature = self.spec_aug(feature)

        if len(list(feature.shape)) == 2:
            feature = np.expand_dims(feature, axis=0) # for other model, need the first 1 dim
        label = self.class_map[self.y_classnames[idx]]

        return feature, label

    def spec_aug(self, x):
        return spec_augment_tensorflow.spec_augment(x,
                                                    time_warping_para=80,
                                                    frequency_masking_para=27,
                                                    time_masking_para=100,
                                                    frequency_mask_num=1,
                                                    time_mask_num=1)

# class Task1aDataSet2020T(Dataset):
#
#     def __init__(self, db_path:str, class_map:dict, feature_folder:str, mode:str="train", device: str = None):
#         #mode: train | test | eval
#         #device：a | b | c
#
#         #read the txt to the data_list
#         self.db_path = db_path
#         self.class_map = class_map
#         df = pd.read_csv("{}/evaluation_setup/fold1_{}.csv".format(db_path, mode), sep="\t")
#
#         # filter out the
#         for index, fp in df.items():
#
#
#         self.X_filepaths = df["filename"].str.replace("audio", feature_folder, n=1).str.replace(".wav", ".npy")
#
#         #TODO: maybe refer to meta.csv
#         if mode == "test":
#             if device is None:
#                 self.y_classnames = df["filename"].str.split("/", expand=True)[1].str.split("-", n=1, expand=True)[0]
#             else:
#                 fp_list = []
#                 y_classnames = []
#                 trg_suffix = "-{}.npy".format(device)
#                 trg_suffix_len = len(trg_suffix)
#                 for index, fp in self.X_filepaths.items():
#                     if not fp[-trg_suffix_len:] == trg_suffix:
#                         continue
#                     fp_list.append(fp)
#                     y_classnames.append(fp.split("/")[1].split("-", 1)[0])
#
#                 self.X_filepaths = fp_list
#                 self.y_classnames = y_classnames
#
#         else:
#             self.y_classnames = df["scene_label"]
#
#
#         self.data_list = []
#
#     def __len__(self):
#         return len(self.X_filepaths)
#
#     def __getitem__(self, idx):
#
#         path = self.X_filepaths[idx]
#         f = open("{}/{}".format(self.db_path, path), 'rb')
#         feature = np.load(f)
#
#         #### no need do swap and squeeze for cnn, code chnaged to accept (1, dim, time)
#         # if feature.shape[0] == 1:
#         #     feature = np.squeeze(feature, axis=0) # for dim256, cnn9
#         # feature = np.swapaxes(feature, 0, 1) # only for cnn
#
#         # feature = self.spec_aug(feature)
#
#         if len(list(feature.shape)) == 2:
#             feature = np.expand_dims(feature, axis=0) # for other model, need the first 1 dim
#         label = self.class_map[self.y_classnames[idx]]
#
#         return feature, label