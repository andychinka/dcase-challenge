import ray
from ray import tune
import os
from hyperopt import hp
from torchvision import transforms

from asc.train import Trainable
from asc.train import TrainStopper

from asc.model.resnet_mod import ResNetMod
from asc.dataset.task1b_dataset_2019 import Task1bDataSet2019
from asc.dataset import transform_utils

exp = ray.tune.Experiment(
            run=Trainable,
            # config={
            #     "network": tune.grid_search(["resnet_mod"]),
            #     "remark": tune.grid_search(["freq_mask=5, time_mask=5, t300, factor=0.7"]),
            #     "optimizer": tune.grid_search(["AdamW"]),
            #     "lr": tune.grid_search([0.0001]),
            #     # weight_decay == 0.1 is very bad
            #     "weight_decay": tune.grid_search([0]),
            #     "momentum": None,
            #     "batch_size": tune.grid_search([32]),
            #     "mini_batch_cnt": 1, # actually batch_size = 256/16 = 16
            #     "mixup_alpha": tune.grid_search([0]),
            #     "mixup_concat_ori": tune.grid_search([False]),
            #
            #     "temporal_crop_length": tune.grid_search([300]),
            #
            #     "specaug_freq_mask": tune.grid_search([True]),
            #     "specaug_freq_mask_args": tune.grid_search([{"num_masks": 5}]),
            #     "specaug_time_mask": tune.grid_search([True]),
            #     "specaug_time_mask_args": tune.grid_search([{"num_masks": 5}]),
            #
            #     # "temporal_crop_length": tune.grid_search([400]),
            #     "feature_folder": tune.grid_search(["logmel_delta2_128_44k"]),
            #     "db_path": os.getenv("HOME") + "/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development",
            #     "model_cls": ResNetMod,
            #     "model_args": {
            #         "out_kernel_size": (132,23)
            #     },
            #     # "composed_transform": transforms.Compose([
            #     #     transform_utils.Normalizer()
            #     # ]),
            #     "data_set_cls": Task1bDataSet2019,
            #     "test_fn": None,  # no use here
            #     # "resume_model": os.getenv("HOME") + "/dcase/dev/ray_results/2019_diff_net_report/Trainable_0_batch_size=32,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0,mixup_concat_ori=False,network=resnet_mod,o_2020-10-07_23-14-22_109tcpy/checkpoint_111/model.pth",
            # },
            name="2019_diff_net_report",
            num_samples=60,
            local_dir=os.getenv("HOME") + "/dcase/result/ray_results",
            stop=TrainStopper(max_ep=200, stop_thres=200),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
            resources_per_trial={"gpu": 1},
        )

hp_space = {
        "network": hp.choice("network", (["resnet_mod"])),
        "optimizer": hp.choice("optimizer", ["AdamW"]),
        "lr": hp.choice("lr", [0.0001]),
        "weight_decay": hp.uniform("weight_decay", 0.001, 0.01),
        "momentum": None,
        "batch_size": 32,
        "mini_batch_cnt": 1,
        "mixup_alpha": 0,
        "mixup_concat_ori": False,
        "feature_folder": "logmel_delta2_128_44k",
        "temporal_crop_length": hp.choice("temporal_crop_length", [300]),

        "specaug_freq_mask": True,
        "specaug_freq_mask_args_nmasks": hp.quniform("specaug_freq_mask_args_nmasks", 1, 10, 1),
        "specaug_time_mask": True,
        "specaug_time_mask_args_nmasks": hp.quniform("specaug_time_mask_args_nmasks", 1, 10, 1),
        # "specaug_time_mask_args": tune.grid_search([{"num_masks": 5}]),

        "model_cls": ResNetMod,
        "model_args": {
            "out_kernel_size": (132,23)
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
        t._train()
        exit()

    ray.shutdown()
    ray.init(local_mode=True, dashboard_host="0.0.0.0", num_cpus=2) # num_cpus limited the cpu assign to ray, default will use all

    algo = HyperOptSearch(
        hp_space,
        metric="acc",
        mode="max",
        n_initial_points=5,
        max_concurrent=1) # make the expierment will add new trial one after one

    hyperopt_h = os.getenv("HOME") + "/dcase/dev/ray_results/2019_diff_net_report/Trainable_1_batch_size=32,feature_folder=logmel_delta2_128_44k,lr=0.0001,mini_batch_cnt=1,mixup_alpha=0,mixup_concat_ori=False,out_2020-10-28_11-18-35ekvaaay3/hyperopt.cp"
    print("hyperopt restor: ", hyperopt_h)
    algo.restore(hyperopt_h)

    import asc.train
    asc.train.set_hyperopt(algo)
    scheduler = AsyncHyperBandScheduler(metric="acc", mode="max", max_t=200)
    analysis = tune.run(
        exp,
        resources_per_trial={"gpu": 1},
        search_alg=algo,
        scheduler=scheduler,
        num_samples=60,
        verbose=1,
        resume=False
    )