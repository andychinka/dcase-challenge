import ray
from ray import tune
from torchvision import transforms
import os

from asc.train import Trainable
from asc.train import TrainStopper

from asc.model.baseline import Baseline
from asc.dataset.task1b_dataset_2019 import Task1bDataSet2019
from asc.dataset import transform_utils

exp = ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["baseline"]),
                "optimizer": tune.grid_search(["AdamW"]),
                "lr": tune.grid_search([0.0001]),
                # weight_decay == 0.1 is very bad
                "weight_decay": tune.grid_search([0]),
                "momentum": None,
                "batch_size": tune.grid_search([32]),
                "mini_batch_cnt": 1, # actually batch_size = 256/16 = 16
                "mixup_alpha": tune.grid_search([0]),
                "mixup_concat_ori": tune.grid_search([False]),
                "feature_folder": tune.grid_search(["logmel_delta2_128_44k"]),
                "db_path": os.getenv("HOME") + "/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development",
                "model_cls": Baseline,
                "model_args": {
                    "maxpool": 84,
                    "full_connected_in": 384,
                    "in_channels": 3,
                },
                "data_set_cls": Task1bDataSet2019,
                "test_fn": None,  # no use here
                "composed_transform": transforms.Compose([
                    transform_utils.Normalizer()
                ]),
                # "resume_model": "/home/hw1-a07/dcase/dev/ray_results/2020_diff_net2/Trainable_0_batch_size=256,feature_folder=mono256dim_norm,lr=0.0001,mixup_alpha=0,mixup_concat_ori=False,network=cnn9avg_amsgrad,o_2020-06-13_11-08-08mq3s_xxl/best_model.pth",
            },
            name="2019_diff_net_report",
            num_samples=1,
            local_dir= os.getenv("HOME") + "/dcase/result/ray_results",
            stop=TrainStopper(max_ep=200, stop_thres=200),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
            resources_per_trial={"gpu": 0, "cpu": 64},
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-test', action='store_true')  # default = false
    args = parser.parse_args()

    if args.test:
        print("====== Test Run =======")
        from asc import exp_utils
        c = exp_utils.exp_to_config(exp)
        t = Trainable(c)
        for e in range(2):
            t._train()
        exit()

    ray.shutdown()
    ray.init(local_mode=True, webui_host="0.0.0.0")

    analysis = tune.run(
        exp,
        verbose=2,
    )