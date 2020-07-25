import ray
from ray import tune

from asc.train import Trainable
from asc.train import TrainStopper

from asc.model.resnet_mod import ResNetMod
from asc.dataset.task1b_dataset_2019 import Task1bDataSet2019

exp = ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["resnet_mod"]),
                "optimizer": tune.grid_search(["SGD"]),
                "scheduler": tune.grid_search(["CosineAnnealingWarmRestarts"]),
                "lr": tune.grid_search([0.1]),
                # weight_decay == 0.1 is very bad
                "weight_decay": tune.grid_search([0]),
                "momentum": tune.grid_search([0.9]),
                "batch_size": tune.grid_search([256]),
                "mini_batch_cnt": 16, # actually batch_size = 256/16 = 16
                "mixup_alpha": tune.grid_search([0.5]),
                "mixup_concat_ori": tune.grid_search([True]),
                "temporal_crop_length": tune.grid_search([400]),
                "feature_folder": tune.grid_search(["logmel_delta2_128_44k"]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development",
                "model_cls": ResNetMod,
                "model_args": {
                    "out_kernel_size": (132,29)
                },
                "data_set_cls": Task1bDataSet2019,
                "test_fn": None,  # no use here
                "smoke_test": False,
            },
            name="2019_diff_net",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(max_ep=200, stop_thres=50),
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

    if not args.test:
        print("====== Test Run =======")
        from asc import exp_utils
        c = exp_utils.exp_to_config(exp)
        t = Trainable(c)
        for _ in range(5):
            t._train()
        exit()

    ray.shutdown()
    ray.init(local_mode=True, webui_host="0.0.0.0")

    analysis = tune.run(
        exp,
        verbose=2,
    )