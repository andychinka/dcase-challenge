import hashlib

import os
import json
import pandas as pd

import matplotlib.pyplot as plt

def cleanup_error_exp(result_folder, exp_state):
    if not exp_state["runner_data"]["_has_errored"]:
        return False

    trash_fp = "{}/_trash".format(result_folder)
    if not os.path.exists(trash_fp):
        os.mkdir(trash_fp)

    for cp in exp_state["checkpoints"]:
        if cp["logdir"] is None:
           continue
        trial_folder = cp["logdir"].split("/")[-1]
        trial_folder_abs = "{}/{}".format(result_folder, trial_folder)
        if os.path.exists(trial_folder_abs):
            print("remove the trial folder: {}".format(trial_folder))
            os.rename(trial_folder_abs, "{}/_trash/{}".format(result_folder, trial_folder))

    #remove exp file
    exp_fp_abs = exp_state["runner_data"]["checkpoint_file"]
    exp_fp = exp_fp_abs.split("/")[-1]
    print("remove exp file", exp_fp)
    os.rename("{}/{}".format(result_folder, exp_fp), "{}/_trash/{}".format(result_folder, exp_fp))

    return True

def gen_report(result_folder: str):
    result_folder_rel = result_folder.split("/")[-1]
    exp_state_fps = []

    # get the exp state file list
    with os.scandir(result_folder) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file() and entry.name.endswith(".json"):
                exp_state_fps.append("{}/{}".format(result_folder, entry.name))

    reports = []
    for fp in exp_state_fps:
        with open(fp) as f:
            exp_state = json.load(f)

            if cleanup_error_exp(result_folder, exp_state):
                continue

            for cp in exp_state["checkpoints"]:
                config = cp["config"]

                # trial info
                logdir = cp["logdir"]
                metric_analysis = cp["metric_analysis"]
                if not metric_analysis:
                    print("no metrix, skip it", cp["trial_id"], fp)
                    continue

                id = hashlib.md5(logdir.encode()).hexdigest()
                network = None if "network" not in config else config["network"]
                feature = config["feature_folder"]

                # hyper params
                # optimizer = None if "optimizer" not in config else config["optimizer"]
                # weight_decay = None if "weight_decay" not in config else  config["weight_decay"]
                # lr = config["lr"]
                # batch_size = config["batch_size"]
                # mixup_alpha = config["mixup_alpha"]
                # mixup_concat_ori = config["mixup_concat_ori"]



                max_acc = metric_analysis["acc"]["max"]
                num_ep = metric_analysis["training_iteration"]["max"]

                #plot grpah for acc and loss
                progresses = pd.read_csv('{}/{}/progress.csv'.format(result_folder, logdir.split("/")[-1])).to_dict()

                plot_loss_acc(
                    list(progresses["train_loss"].values()),
                    list(progresses["val_loss"].values()),
                    list(progresses["acc"].values()),
                    "report/{}_{}_loss_acc.png".format(result_folder.split("/")[-1], id)
                )

                report = {
                    "id": id,
                    "network": network,
                    "feature": feature,
                    "num_ep": "<label title='" + logdir + "'>" + str(num_ep) + "</label>",
                    # "optimizer": optimizer,
                    # "weight_decay": weight_decay,
                    # "lr": lr,
                    # "batch_size": batch_size,
                    # "mixup_alpha": mixup_alpha,
                    # "mixup_concat_ori": mixup_concat_ori,
                    "max_acc": max_acc,
                    "plot_path": "<img style='width: 400px' src='{}_{}_loss_acc.png'/>".format(result_folder_rel, id),
                }

                ignore_cols = ["feature_folder", "db_path", "model_save_fp", "model_cls", "model_args", "data_set_cls", "test_fn"]
                for key in config:
                    if key in ignore_cols:
                        continue
                    report[key] = config[key]
                reports.append(report)

    with open('report/{}.json'.format(result_folder_rel), 'w') as fp:
        json.dump(reports, fp)

    df = pd.DataFrame.from_dict(reports)
    df.to_csv('report/{}.csv'.format(result_folder_rel))
    df.to_html('report/{}.html'.format(result_folder_rel), escape=False, bold_rows=False, table_id=None)

    with open('report/{}.html'.format(result_folder_rel), "a") as file_object:
        file_object.write("""        
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.js"></script>
        
        <script>
          $(document).ready( function () {
            var table = $('table').DataTable({
                paging: false,
                columnDefs: [
                    {
                        targets: [1],
                        visible: false,
                        searchable: false
                    }   
                ],
                initComplete: function () {
                    this.api().columns().every( function () {
                        var column = this;
                        var select = $('<select><option value=""></option></select>')
                            .appendTo( $(column.header()) )
                            .on( 'change', function () {
                                var val = $.fn.dataTable.util.escapeRegex(
                                    $(this).val()
                                );
         
                                column
                                    .search( val ? '^'+val+'$' : '', true, false )
                                    .draw();
                            } );
         
                        column.data().unique().sort().each( function ( d, j ) {
                            select.append( '<option value="'+d+'">'+d+'</option>' )
                        } );
                    } );
                }
            });

            //highlight
            $('tbody').on( 'mouseenter', 'td', function () {
              var colIdx = table.cell(this).index().column;
              $( table.cells().nodes() ).removeClass( 'highlight' );
              $( table.column( colIdx ).nodes() ).addClass( 'highlight' );
            });
        });
        </script>
        <style>
          td.highlight {
              background-color: whitesmoke !important;
          }
        </style>
        """)

    df.to_html('report/{}-pivot.html'.format(result_folder_rel), escape=False, bold_rows=False, table_id="raw-table")
    with open('report/{}-pivot.html'.format(result_folder_rel), "a") as file_object:
        file_object.write("""
        <div id="pivot-table"></div>
        
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.23.0/pivot.min.js"></script>
        <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/pivottable/2.23.0/pivot.min.css">
        <script src="https://code.jquery.com/ui/1.12.0/jquery-ui.min.js"></script>
        
        <style>
            #raw-table{
                display: none;
            }
        </style>
        <script>
            $(function(){
                $("#pivot-table").pivotUI($("#raw-table"),{
                    rows: ["network", "feature"],
                    cols: ["optimizer", "weight_decay"],
                    vals: ["max_acc"],
                    aggregatorName: "Median",
                    rendererName: "Table Barchart"
                });
            });
        </script>
        """)

    print(reports)


def plot_loss_acc(train_losses_list, eval_losses_list, acc_list, save_fp):
    handles = []
    fig, axs = plt.subplots(1, 2)

    eval_losses, = axs[0].plot(eval_losses_list, label="Eval Loss")
    train_losses, = axs[0].plot(train_losses_list, label="Train Loss")
    handles.append(train_losses)
    handles.append(eval_losses)

    #Loss
    axs[0].legend(handles=handles)
    axs[0].grid(True)
    axs[0].set_title(" Loss / epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel('Loss')

    #Acc
    axs[1].plot(acc_list, label="Acc")
    axs[1].grid(True)
    axs[1].set_title("Acc")

    plt.savefig(save_fp)
    plt.close()

if __name__ == "__main__":

    gen_report("../ray_results/2020_diff_net2")