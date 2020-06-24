import ray
from ray import tune

from asc.train import Trainable
from asc.train import TrainStopper

from asc.model import cnn
from asc.model import vgg
from asc.model.alexnet import AlexNet
from asc.dataset.task1a_dataset_2020 import Task1aDataSet2020

exp = ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["cnn9avg_amsgrad"]),
                "optimizer": tune.grid_search(["Adam"]),
                "lr": tune.grid_search([0.0001]),
                # weight_decay == 0.1 is very bad
                "weight_decay": tune.grid_search([0]),
                "momentum": None,
                # "momentum": tune.grid_search([0, 0.1, 0.5, 0.9]),
                "batch_size": tune.grid_search([256]),
                "mini_batch_cnt": 16, # actually batch_size = 256/16 = 16
                "mixup_alpha": tune.grid_search([0]),
                "mixup_concat_ori": tune.grid_search([False]),
                "feature_folder": tune.grid_search(["openl3-music-mel256-emb512-hop0_1"]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "model_cls": cnn.Cnn_9layers_AvgPooling,
                "model_args": {
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
                # "resume_model": "/home/hw1-a07/dcase/dev/ray_results/2020_diff_net2/Trainable_0_batch_size=256,feature_folder=mono256dim_norm,lr=0.0001,mixup_alpha=0,mixup_concat_ori=False,network=cnn9avg_amsgrad,o_2020-06-13_11-08-08mq3s_xxl/best_model.pth",
            },
            name="2020_diff_net2",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(max_ep=200),
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