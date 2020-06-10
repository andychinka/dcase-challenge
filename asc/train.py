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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# device = torch.device("cpu")

def evaluate(model, dataloader):
    model.eval()

    total_loss = 0
    batch = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    confusion_matrix = np.zeros((10, 10), dtype=int) # 2D, [actual_cls][predicted_cls]

    with torch.no_grad():
        for x, targets in dataloader:
            x = torch.FloatTensor(x).to(device)
            targets = torch.LongTensor(targets).to(device)
            outputs = model(x)
            loss = model.cal_loss(outputs, targets)
            total_loss += loss.item()

            outputs = F.log_softmax(outputs, dim=-1)
            _, predicted = torch.max(outputs, 1)

            c = (predicted == targets).squeeze()
            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
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
    print("Overall Acc: {}".format(acc))
    for i in range(10):
        if class_total[i] == 0:
            continue
        print('Accuracy of {} : {}%'.format(i, 100 * class_correct[i] / class_total[i]))

    return (total_loss / batch), acc, class_correct, class_total, confusion_matrix


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
        self.reset_config(c)

    def _train(self):  # This is called iteratively.

        self.model.train()
        # batch = 0
        total_loss = 0

        for param_group in self.optimizer.param_groups:
            self.current_lr = param_group['lr']

        for batch, (x, targets) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            inputs, targets_a, targets_b, lam = data_aug.mixup_data(x, targets,
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

            # batch += 1

        self.train_losses.append(total_loss / batch)

        # save check point
        # torch.save({
        #     "ep": current_ep,
        #     "train_losses": train_losses,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        # }, model_save_fp.format("cp"))

        # evaluation here
        eval_loss, acc, class_correct, class_total, confusion_matrix = evaluate(self.model, self.dataloader_eval)
        self.eval_losses.append(eval_loss)
        print("eval loss: {}, acc: {}".format(eval_loss, acc))

        # plot_loss(self.train_losses, self.eval_losses, self.current_ep, self.model_save_fp)

        if acc > self.best_acc:
            self.not_improve_cnt = 0
            self.best_acc = acc
            print("best model found! save it.")
            # store best model
            # torch.save({
            #     "ep": self.current_ep,
            #     "train_losses":self.train_losses,
            #     "eval_loss": eval_loss,
            #     "acc": self.best_acc,
            #     "class_correct": class_correct,
            #     "class_total": class_total,
            #     "model_state_dict": self.model.state_dict(),
            #     "optimizer_state_dict": self.optimizer.state_dict(),
            # }, self.model_save_fp.format("best"))
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
            print("saving to ", best_model_path)
            self.not_improve_cnt = 0
            print("best model found! save it.")
            # store best model
            torch.save({
                "ep": self.current_ep,
                "acc": self.best_acc,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
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

    def reset_config(self, c):
        print("_______ reset_config")

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
        momentum = None if "momentum" not in c else c["momentum"]

        data_set = data_set_cls(db_path, config.class_map, feature_folder=feature_folder)
        data_set_eval = data_set_cls(db_path, config.class_map, feature_folder=feature_folder, mode="evaluate")

        max_ep = 100
        self.losses = []
        self.train_losses = []
        self.eval_losses = []
        self.log_interval = 10
        self.early_stop_thres = 20
        self.start_time = time.time()

        self.best_acc = 0
        self.previous_acc = 0
        self.not_improve_cnt = 0

        self.current_lr = self.lr
        self.current_ep = 0

        self.model = model_cls(**model_args).to(device)
        print(self.model)
        # summary(self.model, (40, 500))
        #weight -1 -> -6 (aggressive)
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
                nesterov=momentum>0
            )
        else:
            raise Exception("Unkown optimizer: {}".format(optimizer))

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)

        self.dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        self.dataloader_eval = DataLoader(data_set_eval, batch_size=batch_size, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        return True


def train(c):

    db_path= c["db_path"]
    feature_folder = c["feature_folder"]
    model_save_fp = c["model_save_fp"]
    model_cls = c["model_cls"]
    model_args = c["model_args"]
    data_set_cls = c["data_set_cls"]
    test_fn = c["test_fn"]
    batch_size = c["batch_size"]
    lr = c["lr"]
    mixup_alpha = c["mixup_alpha"]

    data_set = data_set_cls(db_path, config.class_map, feature_folder=feature_folder)
    data_set_eval = data_set_cls(db_path, config.class_map, feature_folder=feature_folder, mode="evaluate")


    max_ep = 100
    losses = []
    train_losses = []
    eval_losses = []
    log_interval = 10
    early_stop_thres = 20
    start_time = time.time()

    best_acc = 0
    previous_acc = 0
    not_improve_cnt = 0

    current_lr = lr
    model = model_cls(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    dataloader_eval = DataLoader(data_set_eval, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    for current_ep in range(max_ep):
        model.train()
        batch = 0
        total_loss = 0
        ep_start_time = time.time()

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        for x, targets in dataloader:
            # if not x.shape == (1, 1, 40, 500):
            #     continue
            inputs, targets_a, targets_b, lam = data_aug.mixup_data(x, targets,
                                                           mixup_alpha, torch.cuda.is_available())
            # inputs, targets_a, targets_b = map(Variable, (inputs,
            #                                               targets_a, targets_b))

            inputs = torch.FloatTensor(inputs).to(device)
            outputs = model(inputs)
            loss = data_aug.mixup_criterion(criterion, outputs, targets_a.to(device), targets_b.to(device), lam)

            # x = torch.FloatTensor(x).to(device)
            # targets = torch.LongTensor(targets).to(device)
            # loss = model.get_loss(x, targets)

            losses.append(loss.item())
            total_loss += loss.item()
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % log_interval == 0 and batch > 0:
                avg_loss = total_loss / batch
                print('| epoch {:3d} | {:5d} batches | '
                      'loss {:5.2f} | lr {}'.format(current_ep, batch, avg_loss, current_lr))

            batch += 1

        train_losses.append(total_loss / batch)

        # save check point
        # torch.save({
        #     "ep": current_ep,
        #     "train_losses": train_losses,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        # }, model_save_fp.format("cp"))

        # evaluation here
        eval_loss, acc, class_correct, class_total, confusion_matrix = evaluate(model, dataloader_eval)
        eval_losses.append(eval_loss)
        print("eval loss: {}, acc: {}".format(eval_loss, acc))

        tune.track.log(acc=acc, train_loss=(total_loss / batch), val_loss=eval_loss, lr=current_lr)

        print("time used for {} ep: {}".format(current_ep, time.time() - ep_start_time))

        plot_loss(train_losses, eval_losses, current_ep, model_save_fp)

        if acc > best_acc:
            not_improve_cnt = 0
            best_acc = acc
            print("best model found! save it.")
            # store best model
            torch.save({
                "ep": current_ep,
                "train_losses": train_losses,
                "eval_loss": eval_loss,
                "acc": acc,
                "class_correct": class_correct,
                "class_total": class_total,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, model_save_fp.format("best"))
        else:
            not_improve_cnt += 1

        # if acc < previous_acc:
        #     lr = lr * 0.9
        #     print("Acc. decrease, reduce lr to ", lr)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        scheduler.step(eval_loss)

        previous_acc = acc

        if not_improve_cnt >= early_stop_thres:
            print("Early stop triggered! model does not improved for {} epochs".format(not_improve_cnt))
            break

    # load the best model
    best_cp = torch.load(model_save_fp.format("best"))
    best_model = model_cls(**model_args).to(device)
    best_model.load_state_dict(best_cp["model_state_dict"])

    test_fn(best_model, db_path, feature_folder, model_save_fp)

    print("Done. Time used: ", time.time() - start_time)

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


    exp = {
        "xception_lr_mixup": ray.tune.Experiment(
            run=Trainable,
            config={
                    "lr": tune.grid_search([0.001, 0.005]),
                    "batch_size": args.batch_size,
                    # "mixup_alpha": 0.5,
                    "mixup_alpha": tune.grid_search([0, 0.5]),
                    "db_path": args.db_path,
                    "feature_folder": args.feature_folder,
                    "model_save_fp": args.model_save_fp,

                    "model_cls": model_cls,
                    "model_args": model_args,
                    "data_set_cls": data_set_cls,
                    "test_fn": test_fn,
            },
            name="xception_lr_mixup",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        "xception_mixup_with_ori": ray.tune.Experiment(
            run=Trainable,
            config={
                "lr": tune.grid_search([0.005]),
                # "lr": tune.loguniform(0.0001, 0.1),
                "batch_size": 128,
                # "mixup_alpha": 0.5,
                "mixup_alpha": tune.grid_search([0.7]),
                "mixup_concat_ori": tune.grid_search([True]),
                "db_path": args.db_path,
                "feature_folder": args.feature_folder,
                "model_save_fp": args.model_save_fp,

                "model_cls": model_cls,
                "model_args": model_args,
                "data_set_cls": data_set_cls,
                "test_fn": test_fn,
            },
            name="xception_mixup_with_ori",
            num_samples=1,
            local_dir="/media/data/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        "baseline_lr_bs_mixup": ray.tune.Experiment(
            run=Trainable,
            config={
                "lr": tune.grid_search([0.0001, 0.0005, 0.001, 0.005, 0.01]),
                "batch_size": tune.grid_search([64, 128, 256]),
                # "mixup_alpha": 0.5,
                "mixup_alpha": tune.grid_search([0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "db_path": args.db_path,
                "feature_folder": args.feature_folder,
                "model_save_fp": args.model_save_fp,

                "model_cls": Baseline,
                "model_args": {},
                "data_set_cls": data_set_cls,
                "test_fn": test_fn,
            },
            name="model_mixup",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        # "xception_mixup_dim256_norm": ray.tune.Experiment(
        #     run=Trainable,
        #     config={
        #         "lr": tune.grid_search([0.0001, 0.0005]),
        #         # "lr": tune.loguniform(0.0001, 0.1),
        #         "batch_size": tune.grid_search([16]),
        #         "mixup_alpha": tune.grid_search([0.5]),
        #         "mixup_concat_ori": tune.grid_search([False]),
        #         "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
        #         "feature_folder": tune.grid_search(["mono256dim/norm"]),
        #         "model_save_fp": args.model_save_fp,
        #
        #         "model_cls": Xception,
        #         "model_args": {
        #             "num_classes": 10,
        #             "in_channel": 1,
        #         },
        #         "data_set_cls": Task1aDataSet2020,
        #         "test_fn": None, #no use here
        #     },
        #     name="2020_xception",
        #     num_samples=1,
        #     local_dir="/home/hw1-a07/dcase/result/ray_results",
        #     stop=TrainStopper(),
        #     checkpoint_freq=1,
        #     keep_checkpoints_num=1,
        #     checkpoint_at_end=True,
        #     checkpoint_score_attr="acc",
        # ),
        "xception_mixup_dim40": ray.tune.Experiment(
            run=Trainable,
            config={
                "lr": tune.grid_search([0.0001, 0.0005]),
                # "lr": tune.loguniform(0.0001, 0.1),
                "batch_size": tune.grid_search([128]),
                "mixup_alpha": tune.grid_search([0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder": tune.grid_search(["mono40dim"]),
                "model_save_fp": args.model_save_fp,

                "model_cls": Xception,
                "model_args": {
                    "num_classes": 10,
                    "in_channel": 1,
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
            },
            name="2020_xception",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        "baseline_mixup_dim40_swap": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["baseline_swap"]),
                "lr": tune.grid_search([0.0001, 0.0005]),
                # "lr": tune.loguniform(0.0001, 0.1),
                "batch_size": tune.grid_search([128]),
                "mixup_alpha": tune.grid_search([0, 0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder": tune.grid_search(["mono40dim"]),
                "model_save_fp": args.model_save_fp,

                "model_cls": Baseline,
                "model_args": {
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
            },
            name="2020_baseline",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        "alexnet_mixup_dim40": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["AlexNet"]),
                "lr": tune.grid_search([0.0001]),
                # "lr": tune.loguniform(0.0001, 0.1),
                "batch_size": tune.grid_search([128]),
                "mixup_alpha": tune.grid_search([1]),
                "mixup_concat_ori": tune.grid_search([True]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder": tune.grid_search(["mono40dim"]),
                "model_save_fp": args.model_save_fp,
                "model_cls": AlexNet,
                "model_args": {
                    "num_classes": 10,
                    "in_channel": 1,
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
        ),
        "alexnet_mixup_dim40_swap": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["AlexNet"]),
                "lr": tune.grid_search([0.0001, 0.0005]),
                # "lr": tune.loguniform(0.0001, 0.1),
                "batch_size": tune.grid_search([128]),
                "mixup_alpha": tune.grid_search([0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder": tune.grid_search(["mono40dim"]),
                "model_save_fp": args.model_save_fp,
                "model_cls": AlexNet,
                "model_args": {
                    "num_classes": 10,
                    "in_channel": 1,
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
        ),
        "alexnet_mixup_dim256": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["AlexNet"]),
                "lr": tune.grid_search([0.0001]),
                # "lr": tune.loguniform(0.0001, 0.1),
                "batch_size": tune.grid_search([64]),
                "mixup_alpha": tune.grid_search([0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder": tune.grid_search(["mono256dim/norm"]),
                "model_save_fp": args.model_save_fp,
                "model_cls": AlexNet,
                "model_args": {
                    "num_classes": 10,
                    "in_channel": 1,
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
            },
            name="2020_diff_net",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        "vgg11bn_mixup_dim40": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["VGG11-bn"]),
                "lr": tune.grid_search([0.0001, 0.0005]),
                # "lr": tune.loguniform(0.0001, 0.1),
                "batch_size": tune.grid_search([128]),
                "mixup_alpha": tune.grid_search([0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder":tune.grid_search(["mono40dim"]),
                "model_save_fp": args.model_save_fp,
                "model_cls": vgg.vgg11_bn,
                "model_args": {
                    "num_classes": 10,
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
            },
            name="2020_diff_net",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        "vgg11_mixup_dim40": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["VGG11"]),
                "lr": tune.grid_search([0.0001, 0.0005]),
                # "lr": tune.loguniform(0.0001, 0.1),
                "batch_size": tune.grid_search([128]),
                "mixup_alpha": tune.grid_search([0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder": tune.grid_search(["mono40dim"]),
                "model_save_fp": args.model_save_fp,
                "model_cls": vgg.vgg11,
                "model_args": {
                    "num_classes": 10,
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
            },
            name="2020_diff_net",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        "xception_mixup_dim256_norm": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["Xception"]),
                "lr": tune.grid_search([0.0001]),
                "batch_size": tune.grid_search([16]),
                "mixup_alpha": tune.grid_search([0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder": tune.grid_search(["mono256dim/norm"]),
                "model_save_fp": args.model_save_fp,
                "model_cls": Xception,
                "model_args": {
                    "num_classes": 10,
                    "in_channel": 1,
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
            },
            name="2020_diff_net",
            num_samples=1,
            local_dir="/home/hw1-a07/dcase/result/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        "cnn_dim40_norm": ray.tune.Experiment(
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
                "model_save_fp": args.model_save_fp,
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
        ),
        "cnn_dim256_norm": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["cnn9avg_amsgrad"]),
                "lr": tune.grid_search([0.0001]),
                "batch_size": tune.grid_search([16]),
                "mixup_alpha": tune.grid_search([0, 1]),
                "mixup_concat_ori": tune.grid_search([False, True]),
                "feature_folder": tune.grid_search(["mono256dim/norm"]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "model_save_fp": args.model_save_fp,
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
        ),
        "local_cnn_dim40_norm": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["cnn9avg_amsgrad"]),
                "lr": tune.grid_search([0.0001]),
                "batch_size": tune.grid_search([64]),
                "mixup_alpha": tune.grid_search([0, 0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "feature_folder": tune.grid_search(["mono40dim"]),
                "db_path": "/Users/andycheung/Downloads/dcase/dataset/TAU-urban-acoustic-scenes-2020-mobile-development",
                "model_save_fp": "/Users/andycheung/Downloads",
                "model_cls": cnn.Cnn_9layers_AvgPooling,
                "model_args": {
                    "classes_num": 10,
                    "activation": 'logsoftmax',
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
            },
            name="2020_diff_net",
            num_samples=1,
            local_dir="/Users/andycheung/Downloads/ray_results",
            stop=TrainStopper(),
            checkpoint_freq=1,
            keep_checkpoints_num=1,
            checkpoint_at_end=True,
            checkpoint_score_attr="acc",
        ),
        "cnn_dim40_norm_swap": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["cnn9avg_amsgrad_swap_remove activate"]),
                "lr": tune.grid_search([0.0001]),
                "batch_size": tune.grid_search([64]),
                "mixup_alpha": tune.grid_search([0, 0.5]),
                "mixup_concat_ori": tune.grid_search([False]),
                "feature_folder": tune.grid_search(["mono40dim"]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "model_save_fp": args.model_save_fp,
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
        ),
        "xception_mixup_dim40_swap": ray.tune.Experiment(
            run=Trainable,
            config={
                "network": tune.grid_search(["xception_swap"]),
                "lr": tune.grid_search([0.0001, 0.0005]),
                # "lr": tune.loguniform(0.0001, 0.1),
                "batch_size": tune.grid_search([128]),
                "mixup_alpha": tune.grid_search([1]),
                "mixup_concat_ori": tune.grid_search([False]),
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder": tune.grid_search(["mono40dim"]),
                "model_save_fp": args.model_save_fp,

                "model_cls": Xception,
                "model_args": {
                    "num_classes": 10,
                    "in_channel": 1,
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
        ),
    }

    t = Trainable({
                "network": "cnn9avg_amsgrad",
                "optimizer": "Adam",
                "weight_decay": 0.1,
                "lr": 0.0001,
                "batch_size": 256,
                "mini_batch_cnt": 4, # actually batch_size = 256/4 = 64
                "mixup_alpha": 1,
                "mixup_concat_ori": False,
                "db_path": "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2020-mobile-development",
                "feature_folder": "mono40dim",
                "model_cls": cnn.Cnn_9layers_AvgPooling,
                "model_args": {
                    "classes_num": 10,
                    "activation": 'logsoftmax',
                },
                "data_set_cls": Task1aDataSet2020,
                "test_fn": None,  # no use here
            })


    t._train()

    # import importlib.util
    #
    # spec = importlib.util.spec_from_file_location("exp_config", args.exp_fp)
    # exp_config = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(exp_config)
    #
    # ray.shutdown()
    # ray.init(local_mode=True, webui_host="0.0.0.0")
    # analysis = tune.run(
    #     exp_config.exp,
    #     verbose=2,
    #     resources_per_trial={"gpu": 1},
    #     # scheduler=ray.tune.schedulers.HyperBandScheduler(metric="mean_accuracy", mode="max")
    # )




