import ray
from ray import tune

from asc.train import Trainable
from asc.train import TrainStopper

from asc.model.xception import Xception
from asc.dataset.task1b_dataset_2019 import Task1bDataSet2019

exp = ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["xception"]),
                "optimizer": tune.grid_search(["AdamW"]),
                "lr": tune.grid_search([0.0001]),
                # weight_decay == 0.1 is very bad
                "weight_decay": tune.grid_search([0.001]),
                "momentum": None,
                # "momentum": tune.grid_search([0, 0.1, 0.5, 0.9]),
                "batch_size": tune.grid_search([256]),
                "mini_batch_cnt": 16, # actually batch_size = 256/16 = 16
                "mixup_alpha": tune.grid_search([1]),
                "mixup_concat_ori": tune.grid_search([True]),
                "feature_folder": tune.grid_search(["logmel_40_44k", "logmel_128_44k"]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development",
                "model_cls": Xception,
                "model_args": {
                    "in_channel": 1,
                },
                "data_set_cls": Task1bDataSet2019,
                "test_fn": None,  # no use here
                "resume_model": None,
            },
            name="2019_diff_net",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
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
        for e in range(5):
            t._train()
        exit()

    ray.shutdown()
    ray.init(local_mode=True, webui_host="0.0.0.0")

    analysis = tune.run(
        exp,
        verbose=2,
    )