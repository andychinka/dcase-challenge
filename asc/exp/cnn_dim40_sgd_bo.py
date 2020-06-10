import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch

from asc.train import Trainable
from asc.train import TrainStopper

from asc.model import cnn
from asc.dataset.task1a_dataset_2020 import Task1aDataSet2020




exp = ray.tune.Experiment(
            run=Trainable,
            config={
                "network": "cnn9avg_amsgrad",
                "optimizer": "SGD",
                #"lr": tune.grid_search([0.1]),
                # "weight_decay": tune.grid_search([0.1, 0.0001, 0.000001]),
                # weight_decay == 0.1 is very bad
                "weight_decay": 0.0001,
                #"momentum": 0.5,
                # "momentum": tune.grid_search([0, 0.1, 0.5, 0.9]),
                "batch_size": 64,
                # "mixup_alpha": tune.grid_search([0, 1]),
                "mixup_alpha": 1,
                "mixup_concat_ori": False,
                # "mixup_concat_ori": tune.grid_search([False]),
                "feature_folder": "mono40dim",
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
            name="bo_2020_diff_net",
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
            num_samples=20,
        )

if __name__ == "__main__":
    ray.shutdown()
    ray.init(local_mode=True, webui_host="0.0.0.0")

    alg = BayesOptSearch(
            {"lr": (0.01, 0.5), "momentum": (0, 1)},
            metric="val_loss",
            mode="min",
            utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0},
            verbose=2,
            max_concurrent=2,
            )
    #alg.optimizer.maximize(init_points=2, n_iter=3,)
    analysis = tune.run(
        exp,
        verbose=2,
        resources_per_trial={"gpu": 1},
        # scheduler=ray.tune.schedulers.HyperBandScheduler(metric="mean_accuracy", mode="max")
        search_alg=alg,
        scheduler=AsyncHyperBandScheduler(metric="val_loss", mode="min")
    )