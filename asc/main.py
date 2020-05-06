from asc.model.baseline import Baseline
from asc.dataset.task1b_dataset_2018 import Task1bDataSet2018
from asc.dataset.task1a_dataset_2018 import Task1aDataSet2018
from asc import config
import torch
import time
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataloader):
    model.eval()

    total_loss = 0
    batch = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for x, targets in dataloader:
            x = torch.FloatTensor(x).to(device)
            targets = torch.LongTensor(targets).to(device)
            outputs = model(x)
            loss = model.cal_loss(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()
            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            batch += 1
    acc = np.array(class_correct).sum() / np.array(class_total).sum()
    print("Overall Acc: {}".format(acc))
    for i in range(10):
        if class_total[i] == 0:
            continue
        print('Accuracy of {} : {}%'.format(i, 100 * class_correct[i] / class_total[i]))

    return (total_loss / batch), acc, class_correct, class_total


def test_task1a_2018(model, db_path, feature_folder, model_save_fp):

    data_set_test = Task1aDataSet2018(db_path, config.class_map, feature_folder=feature_folder, mode="test")
    dataloader_test = DataLoader(data_set_test, batch_size=128, shuffle=False)

    test_loss, test_acc, test_class_correct, test_class_total = evaluate(model, dataloader_test)
    print("test acc: {}".format(test_acc))
    torch.save({
        "test_loss": test_loss,
        "test_acc": test_acc,
        "class_correct": test_class_correct,
        "class_total": test_class_total,
    }, model_save_fp.format("test-result"))


def test_task1b_2018(model, db_path, feature_folder, model_save_fp):
    result = {}
    devices = ["a", "b", "c"]

    for d in devices:
        data_set_test = Task1bDataSet2018(db_path, config.class_map, feature_folder=feature_folder, mode="test", device=d)
        dataloader_test = DataLoader(data_set_test, batch_size=128, shuffle=False)
        test_loss, test_acc, test_class_correct, test_class_total = evaluate(model, dataloader_test)
        print("test acc for device-{}: {}".format(device, test_acc))

        result[d] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "class_correct": test_class_correct,
            "class_total": test_class_total,
        }

    torch.save(result, model_save_fp.format("test-result"))


def main(db_path:str, feature_folder: str, model_save_fp:str, data_set_cls, test_fn):

    # Read the meta and prepare for different data set


    data_set = data_set_cls(db_path, config.class_map, feature_folder=feature_folder)
    data_set_eval = data_set_cls(db_path, config.class_map, feature_folder=feature_folder, mode="evaluate")


    max_ep = 100
    batch_size = 128
    losses = []
    log_interval = 10
    early_stop_thres = 5
    start_time = time.time()

    best_acc = 0
    not_improve_cnt = 0

    model = Baseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    dataloader_eval = DataLoader(data_set_eval, batch_size=batch_size, shuffle=False)

    for current_ep in range(max_ep):
        model.train()
        batch = 0
        total_loss = 0
        ep_start_time = time.time()

        for x, targets in dataloader:
            x = torch.FloatTensor(x).to(device)
            targets = torch.LongTensor(targets).to(device)
            loss = model.get_loss(x, targets)

            losses.append(loss.item())
            total_loss += loss.item()
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % log_interval == 0 and batch > 0:
                avg_loss = total_loss / batch
                print('| epoch {:3d} | {:5d} batches | '
                      'loss {:5.2f}'.format(current_ep, batch, avg_loss))

            batch += 1

        # save check point
        torch.save({
            "ep": current_ep,
            "losses": losses,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, model_save_fp.format("cp"))

        # evaluation here
        eval_loss, acc, class_correct, class_total = evaluate(model, dataloader_eval)
        print("eval loss: {}, acc: {}".format(eval_loss, acc))

        print("time used for {} ep: {}".format(current_ep, time.time() - ep_start_time))

        if acc > best_acc:
            not_improve_cnt = 0
            best_acc = acc
            print("best model found! save it.")
            # store best model
            torch.save({
                "ep": current_ep,
                "losses": losses,
                "eval_loss": eval_loss,
                "acc": acc,
                "class_correct": class_correct,
                "class_total": class_total,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, model_save_fp.format("best"))
        else:
            not_improve_cnt += 1

        if not_improve_cnt >= early_stop_thres:
            print("Early stop triggered! model does not improved for {} epochs".format(not_improve_cnt))
            break

    # load the best model
    best_cp = torch.load(model_save_fp.format("best"))
    best_model = Baseline().to(device)
    best_model.load_state_dict(best_cp["model_state_dict"])

    test_fn(best_model, db_path, feature_folder, model_save_fp)

    print("Done. Time used: ", time.time() - start_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-db_path', default="/Users/andycheung/Downloads/dcase/dataset/TUT-urban-acoustic-scenes-2018-mobile-development")
    parser.add_argument('-feature_folder', default="feature")
    parser.add_argument("-model_save_fp", default="./model-{}.pt")

    args = parser.parse_args()

    # main(db_path=args.db_path, feature_folder=args.feature_folder, model_save_fp=args.model_save_fp,
    #      data_set_cls=Task1aDataSet2018, test_fn=test_task1a_2018)

    main(db_path=args.db_path, feature_folder=args.feature_folder, model_save_fp=args.model_save_fp,
         data_set_cls=Task1bDataSet2018, test_fn=test_task1b_2018)


