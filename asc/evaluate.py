import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from asc.train import evaluate
from asc import config

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-db_path',
                        default="/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development")
    parser.add_argument("-model_fp", default="/Users/andycheung/Downloads/dcase-result/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0.5,mixup_concat_ori=True,network=resnet_mod_2020-07-14_13-47-567dphtomu")
    args = parser.parse_args()

    # read param json
    param = pickle.load(open(args.model_fp + "/params.pkl", "rb"))

    # prepare model
    model = param["model_cls"](**param["model_args"]).to(device)
    cp = torch.load(args.model_fp + "/best_model.pth", map_location=device)
    model.load_state_dict(cp["model_state_dict"])

    # prepare dataset
    data_set_eval = param["data_set_cls"](db_path, config.class_map, feature_folder=param["feature_folder"], mode="evaluate")
    dataloader_eval = DataLoader(data_set_eval, batch_size=1, shuffle=False)

    eval_loss, acc, class_correct, class_total, confusion_matrix, raw = evaluate(model, dataloader_eval)