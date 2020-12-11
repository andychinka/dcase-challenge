import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from asc.dataset.task1b_dataset_2019_test import Task1bDataSet2019Test
from asc.config import class_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
This inference format is basic on kaggle required submission format
"""
def inference(db_path, model_path, model_cls, model_args, output_csv_fp):

    model = model_cls(**model_args).to(device)

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
    df.to_csv(output_csv_fp, sep=',', index=True, index_label="Id")


if __name__ == "__main__":
    from asc.model.resnet_mod import ResNetMod
    from asc.model import cnn

    # TODO: hanlding diff model, dataset

    db_path = "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-leaderboard"
    models = [
        # {
        #     "model_path": "/home/hw1-a07/dcase/dev/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0.5,mixup_concat_ori=True,network=cnn9avg_se_2020-07-22_09-01-11_cp1gzsx/best_model.pth",
        #     "model_cls": cnn.Cnn_9layers_AvgPooling_SepFreq,
        #     "model_args": {
        #         "in_channel": 3,
        #         "classes_num": 10,
        #         "activation": 'logsoftmax',
        #         "permute": True,
        #     },
        #     "output_csv_fp": "./predict_cnn_sepfreq_2020-07-22_09-01-11_cp1gzsx.csv"
        # },
        # {
        #     "model_path": "/home/hw1-a07/dcase/dev/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0.5,mixup_concat_ori=True,network=cnn9avg_am_2020-07-21_13-52-40p0oh9xl4/best_model.pth",
        #     "model_cls": cnn.Cnn_9layers_AvgPooling,
        #     "model_args": {
        #         "in_channel": 3,
        #         "classes_num": 10,
        #         "activation": 'logsoftmax',
        #         "permute": True,
        #     },
        #     "output_csv_fp": "./predict_cnn_2020-07-21_13-52-40p0oh9xl4.csv"
        # },
        # {
        #     "model_path": "/home/hw1-a07/dcase/dev/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0.5,mixup_concat_ori=True,network=cnn9avg_am_2020-07-16_09-24-35ef8wkl7m/best_model.pth",
        #     "model_cls": cnn.Cnn_9layers_AvgPooling,
        #     "model_args": {
        #         "in_channel": 3,
        #         "classes_num": 10,
        #         "activation": 'logsoftmax',
        #         "permute": True,
        #     },
        #     "output_csv_fp": "./predict_cnn_2020-07-16_09-24-35ef8wkl7m.csv"
        # },
        # {
        #     "model_path": "/home/hw1-a07/dcase/dev/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0.5,mixup_concat_ori=True,network=cnn9avg_am_2020-07-16_09-24-35ef8wkl7m/best_model.pth",
        #     "model_cls": cnn.Cnn_9layers_AvgPooling,
        #     "model_args": {
        #         "in_channel": 3,
        #         "classes_num": 10,
        #         "activation": 'logsoftmax',
        #         "permute": True,
        #     },
        #     "output_csv_fp": "./predict_cnn_2020-07-16_09-24-35ef8wkl7m.csv"
        # },
        # {
        #     "model_path": "/home/hw1-a07/dcase/dev/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0.5,mixup_concat_ori=True,network=resnet_mod_2020-07-14_13-47-567dphtomu/best_model.pth",
        #     "model_cls": ResNetMod,
        #     "model_args": {
        #     },
        #     "output_csv_fp": "./predict_resnet_2020-07-14_13-47-567dphtomu.csv"
        # },
        # {
        #     "model_path": "/home/hw1-a07/dcase/dev/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.1,mixup_alpha=0.5,mixup_concat_ori=True,momentum=0.9,network=_2020-08-01_15-48-547vpkemap/best_model.pth",
        #     "model_cls": ResNetMod,
        #     "model_args": {
        #         "out_kernel_size": (132, 29)
        #     },
        #     "output_csv_fp": "./predict_resnet_2020-08-01_15-48-547vpkemap.csv"
        # },
        {
            "model_path": "/home/hw1-a07/dcase/dev/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.1,mixup_alpha=0.5,mixup_concat_ori=True,momentum=0.9,network=_2020-08-05_09-48-306tifm25u/best_model.pth",
            "model_cls": ResNetMod,
            "model_args": {
                "out_kernel_size": (132, 29)
            },
            "output_csv_fp": "./predict_resnet_2020-08-05_09-48-306tifm25u.csv"
        },
        {
            "model_path": "/home/hw1-a07/dcase/dev/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.1,mixup_alpha=0.5,mixup_concat_ori=True,momentum=0.9,network=_2020-08-05_09-48-306tifm25u/checkpoint_300/model.pth",
            "model_cls": ResNetMod,
            "model_args": {
                "out_kernel_size": (132, 29)
            },
            "output_csv_fp": "./predict_resnet_2020-08-05_09-48-306tifm25u_checkpoint300.csv"
        }

    ]
    # model_path = "/home/hw1-a07/dcase/result/ray_results/2019_diff_net/Trainable_0_batch_size=256,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0.5,mixup_concat_ori=True,network=resnet_mod_2020-07-14_13-47-567dphtomu/best_model.pth"
    # model_cls = ResNetMod
    # model_args = {
    #     "out_kernel_size": (132,31)
    # }
    # output_csv_fp = "./predict.csv"
    for m in models:
        model_path = m["model_path"]
        model_cls = m["model_cls"]
        model_args = m["model_args"]
        output_csv_fp = m["output_csv_fp"]
        inference(db_path, model_path, model_cls, model_args, output_csv_fp)
        print(output_csv_fp + " Done.")
    print("All Done.")