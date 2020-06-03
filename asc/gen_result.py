import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from asc import config

def gen_task1a_2018(model_fps: list, output_folder: str):

    """
    test_result:
    {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "class_correct": test_class_correct,
        "class_total": test_class_total,
        "confusion_matrix": confusion_matrix,
    }
    """

    n_classes = len(config.class_map)
    labels_classes = list(config.class_map.keys())
    n_models = len(model_fps)
    acc_matrix = np.zeros((n_models, n_classes))
    confusion_matrix = np.zeros((n_classes, n_classes))

    for i, model_fp in enumerate(model_fps):
        test_result = torch.load(model_fp)

        acc_arr = np.asarray(test_result["class_correct"]) / np.asarray(test_result["class_total"])
        acc_matrix[i] = acc_arr

        confusion_matrix = confusion_matrix + test_result["confusion_matrix"]

    # Avg Accuracy
    acc_avg_by_class = np.average(acc_matrix, axis=0)
    acc_avg_by_class_df = pd.DataFrame(acc_avg_by_class, index=labels_classes, columns=["Avg Accuracy"])

    # Confusion Matrix
    confusion_matrix = confusion_matrix/n_models

    #export to file
    plot_heatmap(confusion_matrix, labels_classes, labels_classes, "Confusion Matrix of Task1A", save_fp="{}/confusion_matrix.png".format(output_folder))
    acc_avg_by_class_df.to_csv("{}/acc_avg_by_class.csv".format(output_folder))
    print(acc_avg_by_class_df)


def gen_task1b_2018(model_fps: list, output_folder: str):

    """
    test_result:
    {
        [device]: {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "class_correct": test_class_correct,
            "class_total": test_class_total,
            "confusion_matrix": confusion_matrix,
        }
    }
    """

    n_classes = len(config.class_map)
    labels_classes = list(config.class_map.keys())
    n_models = len(model_fps)
    devices = ["a", "b", "c"]

    for d in devices:
        acc_matrix = np.zeros((n_models, n_classes))
        confusion_matrix = np.zeros((n_classes, n_classes))

        for i, model_fp in enumerate(model_fps):
            test_result = torch.load(model_fp)
            test_result = test_result[d]
            acc_arr = np.asarray(test_result["class_correct"]) / np.asarray(test_result["class_total"])
            acc_matrix[i] = acc_arr

            confusion_matrix = confusion_matrix + test_result["confusion_matrix"]

        # Avg Accuracy
        acc_avg_by_class = np.average(acc_matrix, axis=0)
        acc_avg_by_class_df = pd.DataFrame(acc_avg_by_class, index=labels_classes, columns=["Avg Accuracy - Device {}".format(d.upper())])

        # Confusion Matrix
        confusion_matrix = confusion_matrix/n_models

        #export to file
        plot_heatmap(confusion_matrix, labels_classes, labels_classes, "Confusion Matrix of Task1B - Device {}".format(d.upper()), save_fp="{}/confusion_matrix-{}.png".format(output_folder, d))
        acc_avg_by_class_df.to_csv("{}/acc_avg_by_class_{}.csv".format(output_folder, d))
        print(acc_avg_by_class_df)


def plot_heatmap(matrix, row_labels, col_labels, title, save_fp=None):
    scale = 6
    fig, ax = plt.subplots(figsize=(scale+2, scale))
    im = ax.imshow(matrix, cmap="Blues")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if matrix[i, j] == 0:
                continue
            text = ax.text(j, i, int(matrix[i, j]),
                           ha="center", va="center")

    ax.set_title(title)
    fig.tight_layout()

    if save_fp is not None:
        plt.savefig(save_fp)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-model_fp_format', default="/media/data/master-proj/2019-1a-baseline-{}-v2-test-result.pt")
    parser.add_argument('-run_cnt', default=10)
    parser.add_argument("-output_folder", default="/media/data/master-proj/2019-1a-v2-baseline-result")
    parser.add_argument("-task", default="task1a-2019")

    args = parser.parse_args()

    result_fn = None
    if args.task in ["task1a-2018", "task1a-2019"]:
        result_fn = gen_task1a_2018
    elif args.task in ["task1b-2018", "task1b-2019"]:
        result_fn = gen_task1b_2018
    else:
        raise Exception("Unknown task: ", args.task)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    model_fps = []
    for i in range(args.run_cnt):
        model_fps.append(args.model_fp_format.format(i+1))

    result_fn(model_fps, args.output_folder)
