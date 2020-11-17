import ray
from ray import tune
import os
from torchvision import transforms

from asc.train import Trainable
from asc.train import TrainStopper

from asc.model.resnet_mod import ResNetMod
from asc.dataset.task1b_dataset_2019 import Task1bDataSet2019
from asc.dataset import transform_utils

exp = ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["resnet_mod"]),
                # "remark": tune.grid_search(["freq_mask=5, time_mask=5"]),
                "optimizer": tune.grid_search(["AdamW"]),
                "lr": tune.grid_search([0.0001]),
                # weight_decay == 0.1 is very bad
                "weight_decay": tune.grid_search([0]),
                "momentum": None,
                "batch_size": tune.grid_search([32]),
                "mini_batch_cnt": 1, # actually batch_size = 256/16 = 16
                "mixup_alpha": tune.grid_search([0]),
                "mixup_concat_ori": tune.grid_search([False]),
                "temporal_crop_length": tune.grid_search([300]),

                "specaug_freq_mask": tune.grid_search([False]),
                # "specaug_freq_mask_args": tune.grid_search([{"num_masks": 5}]),
                "specaug_time_mask": tune.grid_search([False]),
                # "specaug_time_mask_args": tune.grid_search([{"num_masks": 5}]),

                # "temporal_crop_length": tune.grid_search([400]),
                "feature_folder": tune.grid_search(["logmel_delta2_128_44k"]),
                "db_path": os.getenv("HOME") + "/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development",
                "model_cls": ResNetMod,
                "model_args": {
                    "out_kernel_size": (132,23)
                },
                # "composed_transform": transforms.Compose([
                #     transform_utils.Normalizer()
                # ]),
                "data_set_cls": Task1bDataSet2019,
                "test_fn": None,  # no use here
                # "resume_model": os.getenv("HOME") + "/dcase/dev/ray_results/2019_diff_net_report/Trainable_0_batch_size=32,feature_folder=logmel_delta2_128_44k,lr=0.0001,mixup_alpha=0,mixup_concat_ori=False,network=resnet_mod,o_2020-10-07_23-14-22_109tcpy/checkpoint_111/model.pth",
            },
            name="2019_diff_net_report",
            num_samples=1,
            local_dir=os.getenv("HOME") + "/dcase/result/ray_results",
            stop=TrainStopper(max_ep=200, stop_thres=200),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
            resources_per_trial={"gpu": 1},
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
        t._train()
        exit()

    ray.shutdown()
    ray.init(local_mode=False, dashboard_host="0.0.0.0", num_cpus=2) # num_cpus limited the cpu assign to ray, default will use all

    analysis = tune.run(
        exp,
        resources_per_trial={"gpu": 1},
        verbose=1,
    )