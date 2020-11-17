import ray
from ray import tune
from hyperopt import hp
import os

from asc.train import Trainable
from asc.train import TrainStopper

from asc.model.baseline import Baseline
from asc.dataset.task1b_dataset_2019 import Task1bDataSet2019

exp = ray.tune.Experiment(
            run=Trainable,
            # config={
            #     "network": tune.grid_search(["baseline"]),
            #     "optimizer": tune.grid_search(["AdamW"]),
            #     "lr": tune.grid_search([0.0001]),
            #     # weight_decay == 0.1 is very bad
            #     "weight_decay": tune.grid_search([0]),
            #     "momentum": None,
            #     "batch_size": 32, #tune.grid_search([32]),
            #     "mini_batch_cnt": 1, # actually batch_size = 256/16 = 16
            #     "mixup_alpha": 0, # tune.grid_search([0]),
            #     "mixup_concat_ori": False, # tune.grid_search([False]),
            #     "feature_folder": "logmel_40_44k", #tune.grid_search(["logmel_40_44k"]),
            #     "db_path": os.getenv("HOME") + "/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development",
            #     "model_cls": Baseline,
            #     "model_args": {
            #         "full_connected_in": 128
            #     },
            #     "data_set_cls": Task1bDataSet2019,
            #     "test_fn": None,  # no use here
            #     # "resume_model": "/home/hw1-a07/dcase/dev/ray_results/2020_diff_net2/Trainable_0_batch_size=256,feature_folder=mono256dim_norm,lr=0.0001,mixup_alpha=0,mixup_concat_ori=False,network=cnn9avg_amsgrad,o_2020-06-13_11-08-08mq3s_xxl/best_model.pth",
            # },
            # name="2019_diff_net_report",
            name="test_resume",
            num_samples=10,
            local_dir=os.getenv("HOME") + "/dcase/result/ray_results",
            stop=TrainStopper(max_ep=10, stop_thres=10),
            checkpoint_freq=1,
            keep_checkpoints_num=2,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
            resources_per_trial={"gpu": 1},
        )

hp_space = {
        # "network": hp.choice("network", [
        #     {
        #         "type": "baseline",
        #         "full_connected_in": 128
        #     }
        # ]),
        "network": hp.choice("network", (["baseline"])),
        "optimizer": hp.choice("optimizer", ["AdamW"]),
        "lr": hp.loguniform("lr", -10, -1),
        "weight_decay": hp.uniform("momentum", 0.1, 0.9),
        "momentum": None,
        "batch_size": 32,  # tune.grid_search([32]),
        "mini_batch_cnt": 1,  # actually batch_size = 256/16 = 16
        "mixup_alpha": 0,  # tune.grid_search([0]),
        "mixup_concat_ori": False,  # tune.grid_search([False]),
        "feature_folder": "logmel_40_44k",  # tune.grid_search(["logmel_40_44k"]),
        # "db_path": os.getenv("HOME") + "/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development",
        "model_cls": Baseline,
        "model_args": {
            "full_connected_in": 128
        },
        # "data_set_cls": Task1bDataSet2019,
        "test_fn": None,  # no use here
        # "resume_model": "/home/hw1-a07/dcase/dev/ray_results/2020_diff_net2/Trainable_0_batch_size=256,feature_folder=mono256dim_norm,lr=0.0001,mixup_alpha=0,mixup_concat_ori=False,network=cnn9avg_amsgrad,o_2020-06-13_11-08-08mq3s_xxl/best_model.pth",
}

if __name__ == "__main__":
    import argparse
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch

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
    ray.init(local_mode=False, dashboard_host="0.0.0.0", num_cpus=2) # num_cpus limited the cpu assign to ray, default will use all

    algo = HyperOptSearch(
        hp_space,
        metric="acc",
        mode="max",
        n_initial_points=5,
        max_concurrent=1) # make the expierment will add new trial one after one
    scheduler = AsyncHyperBandScheduler(metric="acc", mode="max")
    analysis = tune.run(
        exp,
        resources_per_trial={"gpu": 1},
        search_alg=algo,
        scheduler=scheduler,
        verbose=1,
        resume=True
    )