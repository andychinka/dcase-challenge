from asc.model.baseline import Baseline
from asc.model.xception import Xception
from asc.model.alexnet import AlexNet
from asc.model import vgg
from asc.model import cnn
from asc.dataset.task1b_dataset_2018 import Task1bDataSet2018
from asc.dataset.task1a_dataset_2018 import Task1aDataSet2018
from asc.dataset.task1a_dataset_2019 import Task1aDataSet2019
from asc.dataset.task1b_dataset_2019 import Task1bDataSet2019
from asc.dataset.task1a_dataset_2020 import Task1aDataSet2020
from asc import config
from asc import data_aug
import torch
import torch.nn.functional as F
import time
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# device = torch.device("cpu")

# it's now only support for batch_size = 1, as need to record the acc by different dimension
def evaluate(model, dataloader):
    model.eval()

    total_loss = 0
    batch = 0

    """
    raw:
        scene(label)
        city 
        device
        predicted
        acc
        loss
    """
    raw = []

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    confusion_matrix = np.zeros((10, 10), dtype=int) # 2D, [actual_cls][predicted_cls]
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, targets, cities, devices in dataloader:
            x = torch.FloatTensor(x).to(device)
            targets = torch.LongTensor(targets).to(device)
            outputs = model(x)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            outputs = F.log_softmax(outputs, dim=-1)
            _, predicted = torch.max(outputs, 1)

            c = (predicted == targets).squeeze()

            raw.append({
                "scene": targets[0].item(),
                "city": cities[0],
                "device": devices[0],
                "predicted": predicted[0].item(),
                "loss": loss.item(),
                "acc": 1 if predicted == targets else 0,
            })

            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c.item()
                class_total[label] += 1

                # TODO: check if this confusion matrix implemented wrongly before
                # confusion_matrix[label.item()][predicted[label.item()].item()] += 1
                confusion_matrix[label.item()][predicted[i].item()] += 1
                """
                confusion_matrix[label.item()][predicted[label.item()].item()] += 1
                IndexError: index 9 is out of bounds for dimension 0 with size 8 at time: 1.5905e+09
                """

            batch += 1
    acc = np.array(class_correct).sum() / np.array(class_total).sum()
    print("<================ Evaluation Result ================")
    print("Overall Acc: {}".format(acc))

    # acc by scene
    for i in range(10):
        if class_total[i] == 0:
            continue
        print('Accuracy of {} : {:.2f}% ({:.0f}/{:.0f})'.format(i, 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
    df = pd.DataFrame.from_dict(raw)
    # print(df.groupby(['scene']).mean()[["loss", "acc"]])
    # acc by device
    print(df.groupby(['device']).mean()[["loss", "acc"]])
    # acc by city
    print(df.groupby(['city']).mean()[["loss", "acc"]])
    print("================ Evaluation Result ================>")

    return (total_loss / batch), acc, class_correct, class_total, confusion_matrix, raw

#TODO: move to another package
def test_task1a_2018(model, db_path, feature_folder, model_save_fp):

    data_set_test = Task1aDataSet2018(db_path, config.class_map, feature_folder=feature_folder, mode="test")
    dataloader_test = DataLoader(data_set_test, batch_size=128, shuffle=False)

    test_loss, test_acc, test_class_correct, test_class_total, confusion_matrix = evaluate(model, dataloader_test)
    print("test acc: {}".format(test_acc))
    torch.save({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "class_correct": test_class_correct,
        "class_total": test_class_total,
        "confusion_matrix": confusion_matrix,
    }, model_save_fp.format("test-result"))


#TODO: move to another package
def test_task1b_2018(model, db_path, feature_folder, model_save_fp):
    result = {}
    devices = ["a", "b", "c"]

    for d in devices:
        data_set_test = Task1bDataSet2018(db_path, config.class_map, feature_folder=feature_folder, mode="test", device=d)
        dataloader_test = DataLoader(data_set_test, batch_size=128, shuffle=False)
        test_loss, test_acc, test_class_correct, test_class_total, confusion_matrix = evaluate(model, dataloader_test)
        print("test acc for device-{}: {}".format(d, test_acc))

        result[d] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "class_correct": test_class_correct,
            "class_total": test_class_total,
            "confusion_matrix": confusion_matrix,
        }

    torch.save(result, model_save_fp.format("test-result"))

#TODO: move to another package
def test_task1a_2019(model, db_path, feature_folder, model_save_fp):

    data_set_test = Task1aDataSet2019(db_path, config.class_map, feature_folder=feature_folder, mode="test")
    dataloader_test = DataLoader(data_set_test, batch_size=128, shuffle=False)

    test_loss, test_acc, test_class_correct, test_class_total, confusion_matrix = evaluate(model, dataloader_test)
    print("test acc: {}".format(test_acc))
    torch.save({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "class_correct": test_class_correct,
        "class_total": test_class_total,
        "confusion_matrix": confusion_matrix,
    }, model_save_fp.format("test-result"))

#TODO: move to another package
def test_task1b_2019(model, db_path, feature_folder, model_save_fp):
    result = {}
    devices = ["a", "b", "c"]

    for d in devices:
        data_set_test = Task1bDataSet2019(db_path, config.class_map, feature_folder=feature_folder, mode="test", device=d)
        dataloader_test = DataLoader(data_set_test, batch_size=128, shuffle=False)
        test_loss, test_acc, test_class_correct, test_class_total, confusion_matrix = evaluate(model, dataloader_test)
        print("test acc for device-{}: {}".format(d, test_acc))

        result[d] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "class_correct": test_class_correct,
            "class_total": test_class_total,
            "confusion_matrix": confusion_matrix,
        }

    torch.save(result, model_save_fp.format("test-result"))

class Trainable(tune.Trainable):

    def _setup(self, c):
        db_path = c["db_path"]
        feature_folder = c["feature_folder"]
        # self.model_save_fp = c["model_save_fp"]
        model_cls = c["model_cls"]
        model_args = c["model_args"]
        data_set_cls = c["data_set_cls"]
        self.test_fn = c["test_fn"]
        batch_size = int(c["batch_size"] / c["mini_batch_cnt"])
        self.mini_batch_cnt = c["mini_batch_cnt"]
        self.lr = c["lr"]
        self.mixup_alpha = c["mixup_alpha"]
        self.mixup_concat_ori = c["mixup_concat_ori"]
        weight_decay = c["weight_decay"]
        optimizer = c["optimizer"]
        scheduler = "ReduceLROnPlateau" if "scheduler" not in c else c["scheduler"]
        momentum = None if "momentum" not in c else c["momentum"]
        resume_model = None if "resume_model" not in c else c["resume_model"]
        self.temporal_crop_length = None if "temporal_crop_length" not in c else c["temporal_crop_length"]
        self.smoke_test = False if "smoke_test" not in c else c["smoke_test"]

        data_set = data_set_cls(db_path, config.class_map, feature_folder=feature_folder)
        data_set_eval = data_set_cls(db_path, config.class_map, feature_folder=feature_folder, mode="evaluate")

        self.losses = []
        self.train_losses = []
        self.eval_losses = []
        self.log_interval = 10
        self.early_stop_thres = 50
        self.start_time = time.time()

        self.best_acc = 0
        self.previous_acc = 0
        self.not_improve_cnt = 0

        self.current_lr = self.lr
        self.current_ep = 0

        self.model = model_cls(**model_args).to(device)
        self.best_model_path = None
        self.best_eval_raw = None

        print(self.model)
        # summary(self.model, (40, 500))
        # weight -1 -> -6 (aggressive)
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
                lr=self.lr,
                amsgrad=True)
        elif optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=weight_decay,
                amsgrad=True
            )
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=momentum,
                dampening=0,
                weight_decay=weight_decay,
                nesterov=momentum > 0
            )
        else:
            raise Exception("Unkown optimizer: {}".format(optimizer))

        # reload model
        if resume_model is not None:
            print("==== resume model from {}".format(resume_model))
            cp = torch.load(resume_model)
            self.model.load_state_dict(cp["model_state_dict"])
            self.optimizer.load_state_dict(cp["optimizer_state_dict"])
            self.best_acc = cp["acc"]
            self.previous_acc = cp["acc"]
            self.current_ep = cp["ep"]

        if scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)
        elif scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                  T_0=3, T_mult=2, eta_min=self.lr*1e-4, last_epoch=-1)
        else:
            raise Exception("Unkown scheduler: {}".format(scheduler))

        self.dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        self.dataloader_eval = DataLoader(data_set_eval, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()

    def _train(self):  # This is called iteratively.

        self.model.train()
        # batch = 0
        total_loss = 0

        for param_group in self.optimizer.param_groups:
            self.current_lr = param_group['lr']

        for batch, (x, targets, cities, devices) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            if self.temporal_crop_length:
                inputs = data_aug.temporal_crop(x, self.temporal_crop_length)

            inputs, targets_a, targets_b, lam = data_aug.mixup_data(inputs, targets,
                                                                    self.mixup_alpha,
                                                                    device.type == "cuda",
                                                                    self.mixup_concat_ori)

            inputs = torch.FloatTensor(inputs).to(device)
            outputs = self.model(inputs)
            loss = data_aug.mixup_criterion(self.criterion, outputs, targets_a.to(device), targets_b.to(device), lam)

            # x = torch.FloatTensor(x).to(device)
            # targets = torch.LongTensor(targets).to(device)
            # loss = model.get_loss(x, targets)

            self.losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()

            if (batch + 1) % self.mini_batch_cnt == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if batch % self.log_interval == 0 and batch > 0:
                avg_loss = total_loss / batch
                print('| epoch {:3d} | {:5d} batches | loss {:5.2f} | lr {}'
                      .format(self.current_ep, batch, avg_loss, self.current_lr))

            if self.smoke_test and batch % self.log_interval == 0 and batch > 0:
                print("----------SMOKE TEST ENABLE, finish the ep---------")
                break


        self.train_losses.append(total_loss / batch)

        # save check point
        # torch.save({
        #     "ep": current_ep,
        #     "train_losses": train_losses,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        # }, model_save_fp.format("cp"))

        # evaluation here
        eval_loss, acc, class_correct, class_total, confusion_matrix, raw = evaluate(self.model, self.dataloader_eval)
        self.eval_losses.append(eval_loss)
        print("eval loss: {}, acc: {}".format(eval_loss, acc))

        # plot_loss(self.train_losses, self.eval_losses, self.current_ep, self.model_save_fp)

        if acc > self.best_acc:
            self.not_improve_cnt = 0
            self.best_acc = acc
            print("best model found! save it.")
            self.best_eval_raw = raw
        else:
            self.not_improve_cnt += 1


        self.scheduler.step(eval_loss)

        self.previous_acc = acc
        self.current_ep += 1

        return {
            "mean_accuracy": acc,
            "acc": acc,
            "train_loss": (total_loss / batch),
            "val_loss": eval_loss,
            "lr": self.current_lr,
        }

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save({
            "ep": self.current_ep,
            "train_losses": self.train_losses,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)

        if self.previous_acc == self.best_acc:
            best_model_path = "{}/../best_model.pth".format(tmp_checkpoint_dir)
            self.best_model_path = best_model_path
            print("saving to ", best_model_path)
            self.not_improve_cnt = 0
            print("best model found! save it.")
            # store best model
            torch.save({
                "ep": self.current_ep,
                "acc": self.best_acc,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "eval_raw": self.best_eval_raw,
            }, best_model_path)
        return tmp_checkpoint_dir

    def _restore(self, checkpoint):
        print("______ restore from ", checkpoint)
        cp = torch.load(checkpoint)
        self.model = model_cls(**model_args).to(device)
        self.model.load_state_dict(cp["model_state_dict"])
        self.optimizer.load_state_dict(cp["optimizer_state_dict"])

        self.train_losses = cp["train_losses"]
        self.current_ep = cp["current_ep"]

    def _stop(self):
        print('_____trianable stop')

        # TODO: generate the report for the best model

    def reset_config(self, c):
        print("_______ reset_config")


        return True


def plot_loss(train_losses_list, eval_losses_list, ep, model_save_fp):
    handles = []
    fig, ax = plt.subplots()

    eval_losses, = ax.plot(eval_losses_list, label="Eval Loss")
    train_losses, = ax.plot(train_losses_list, label="Train Loss")
    handles.append(train_losses)
    handles.append(eval_losses)

    ax.legend(handles=handles)
    ax.grid(True)
    ax.set_title(" Loss / epoch @ ep {}".format(ep))
    ax.set_xlabel("Epoch")
    ax.set_ylabel('Loss')

    folder = model_save_fp.split(".pt")[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.savefig("{}/loss-ep.png".format(folder))
    # plt.show()

class TrainStopper(ray.tune.Stopper):

    def __init__(self, stop_thres:int = 10, max_ep = 100):
        self.trial_best = {}
        self.trial_ep_cnt = {}
        self.stop_thres = stop_thres
        self.max_ep = max_ep

    def __call__(self, trial_id, result):
        print("calling stopper", trial_id)
        print(result)

        if trial_id not in self.trial_best:
            self.trial_best[trial_id] = {
                "acc": 0,
                "not_improved_cnt": 0
            }
            self.trial_ep_cnt[trial_id] = 0

        self.trial_ep_cnt[trial_id] += 1
        if self.trial_best[trial_id]["acc"] < result["acc"]:
            self.trial_best[trial_id] = {
                "acc": result["acc"],
                "not_improved_cnt": 0
            }
        else:
            self.trial_best[trial_id]["not_improved_cnt"] += 1

        if self.trial_best[trial_id]["not_improved_cnt"] >= self.stop_thres:
            print("not_improved for ", self.trial_best[trial_id]["not_improved_cnt"], ", stop now")
            return True

        if self.trial_ep_cnt[trial_id] >= self.max_ep:
            print("trail # of ep > max_ep, stop now.")
            return True

        return False

    def stop_all(self):
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-db_path', default="/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development")
    parser.add_argument('-feature_folder', default="feature2")
    parser.add_argument("-model_save_fp", default="/home/hw1-a07/dcase/result/2019-1b-xception-tmp-{}.pt")
    parser.add_argument("-task", default="task1b-2019")
    parser.add_argument("-model", default="xception")
    parser.add_argument("-batch_size", default=128)
    parser.add_argument("-lr", default=0.005)
    parser.add_argument("-exp_fp", default="./exp/cnn_dim40_sgd.py")
    args = parser.parse_args()

    data_set_cls = None
    test_fn = None
    model_cls = None
    model_args = {}

    if args.model == "baseline":
        model_cls = Baseline
    elif args.model == "xception":
        model_cls = Xception
        model_args = {
            "num_classes": 10,
            "in_channel": 1,
        }
    else:
        raise Exception("Unknown model: ", args.model)

    if args.task in ["task1a-2018"]:
        data_set_cls = Task1aDataSet2018
        test_fn = test_task1a_2018
    elif args.task in ["task1b-2018"]:
        data_set_cls = Task1bDataSet2018
        test_fn = test_task1b_2018
    elif args.task in ["task1a-2019"]:
        data_set_cls = Task1aDataSet2019
        test_fn = test_task1a_2019
    elif args.task in ["task1b-2019"]:
        data_set_cls = Task1bDataSet2019
        test_fn = test_task1b_2019
    else:
        raise Exception("Unknown task: ", args.task)

    t = Trainable({
                "network": "baseline",
                "optimizer": "Adam",
                "weight_decay": 0.1,
                "lr": 0.0001,
                "batch_size": 256,
                "mini_batch_cnt": 4, # actually batch_size = 256/4 = 64
                "mixup_alpha": 1,
                "mixup_concat_ori": False,
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development",
                "feature_folder": "feature2",
                "model_cls": Baseline,
                "model_args": {
                },
                "data_set_cls": Task1bDataSet2019,
                "test_fn": None,  # no use here
            })


    t._train()




