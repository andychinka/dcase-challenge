import ray
from ray import tune

from asc.train import Trainable
from asc.train import TrainStopper

from asc.model import cnn
from asc.dataset.task1a_dataset_2020 import Task1aDataSet2020

exp = ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["cnn9avg_amsgrad"]),
                "optimizer": tune.grid_search(["SGD"]),
                "lr": tune.grid_search([0.1]),
                # "weight_decay": tune.grid_search([0.1, 0.0001, 0.000001]),
                # weight_decay == 0.1 is very bad
                "weight_decay": tune.grid_search([0.0001]),
                "momentum": tune.grid_search([0.5]),
                # "momentum": tune.grid_search([0, 0.1, 0.5, 0.9]),
                "batch_size": tune.grid_search([64]),
                # "mixup_alpha": tune.grid_search([0, 1]),
                "mixup_alpha": tune.grid_search([1]),
                "mixup_concat_ori": tune.grid_search([False, True]),
                # "mixup_concat_ori": tune.grid_search([False]),
                "feature_folder": tune.grid_search(["mono40dim"]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                # "model_save_fp": args.model_save_fp,
                "model_cls": cnn.Cnn_9layers_AvgPooling,
                "model_args": {
                    "classes_num": 10,
                    "activation": 'logsoftmax',
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
            },
            name="2020_diff_net2",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        )