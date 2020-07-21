import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from asc.dataset.task1b_dataset_2019_test import Task1bDataSet2019Test
from asc.config import class_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(model_path, model_cls, model_args):

    model = model_cls(**model_args).to(device)
    db_path = "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-leaderboard"

    #Load Model
    cp = torch.load(model_path)
    model.load_state_dict(cp["model_state_dict"])


    #Setup DataSet
    data_set = Task1bDataSet2019Test(db_path)
    dataloader = DataLoader(data_set, batch_size=128, shuffle=False)

    inference_results = []

    #inference
    with torch.no_grad():
        for x in dataloader:
            x = torch.FloatTensor(x).to(device)
            outputs = model(x)
            outputs = F.log_softmax(outputs, dim=-1)
            _, predicted = torch.max(outputs, 1)
            inference_results += predicted.tolist()

    # save to file
    for i, r in enumerate(inference_results):
        inference_results[i] = list(class_map.keys())[r]

    df = pd.DataFrame(data={"Scene_label": inference_results})
    df.to_csv("./predict.csv", sep=',', index=True, index_label="Id")


if __name__ == "__main__":
    from asc.model.resnet_mod import ResNetMod
    model_path = "/home/hw1-a07/dcase/result/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0.5,mixup_concat_ori=True,network=resnet_mod_2020-07-14_13-47-567dphtomu/best_model.pth"
    model_cls = ResNetMod
    model_args = {
        "out_kernel_size": (132,31)
    }
    inference(model_path, model_cls, model_args)