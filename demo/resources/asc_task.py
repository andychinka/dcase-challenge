import time
import os
from os import path
from os import environ
from shutil import copyfile
import matplotlib.pyplot as plt


from asc.preprocess.log_mel_htk_preprocess import LogMelHTKPreProcess
from asc.model.resnet_mod import ResNetMod
from asc.model.baseline import Baseline
from asc import config

import torch
from flask_restful import Resource, fields, marshal_with, marshal, abort, reqparse, request
from flask import current_app as app

print("resmod_cp", environ.get("resmod_cp"))
print("baseline_cp", environ.get("baseline_cp"))

cp1 = torch.load(environ.get("resmod_cp"))
model_args = {"out_kernel_size": (132,23)}
model_resmod = ResNetMod(**model_args)
model_resmod.load_state_dict(cp1["model_state_dict"])

cp = torch.load(environ.get("baseline_cp"))
model_args = {"maxpool": 84, "full_connected_in": 384, "in_channels": 3}
model_baseline = Baseline(**model_args)
model_baseline.load_state_dict(cp["model_state_dict"])


models = {
    "resmod": model_resmod,
    "baseline": model_baseline,
}

class AscTaskListResource(Resource):

    def __init__(self):
        super(AscTaskListResource, self).__init__()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('type', type=str, required=True, choices=["upload", "preset"])
        parser.add_argument('preset_path', type=str, required=False)
        parser.add_argument('upload_path', type=str, required=False)
        parser.add_argument('model_code', type=str, required=True)
        parser.add_argument('feature_code', type=str, required=True)

        args = parser.parse_args()
        tasks_folder = os.path.abspath("demo/static/asc_tasks")

        asc_task_id = str(time.time())

        # Create Task Folder
        if not path.exists(tasks_folder):
            os.mkdir(tasks_folder)
        task_folder = tasks_folder + "/" + asc_task_id
        os.mkdir(task_folder)

        # Save the audio
        if args["type"] == "preset":
            audio_fp = os.getcwd() + "/" + args["preset_path"]
        else: # upload
            audio_fp = os.getcwd() + "/demo/static/upload/" + args["upload_path"]

        copyfile(audio_fp, task_folder + "/input.wav")


        # extract feature
        preProcess = LogMelHTKPreProcess("", "")
        features = preProcess.extract_feature(task_folder + "/input.wav")
        inputs = torch.FloatTensor(features)
        inputs = torch.unsqueeze(inputs, 0)
        print("features.shape", features.shape)

        fig, axs = plt.subplots(3)
        for i, feature in enumerate(features):
            axs[i].imshow(feature)
        plt.savefig(task_folder + "/features.png")

        # predict
        output = models[args["model_code"]](inputs)
        output = torch.nn.functional.softmax(output, dim=1)
        print("output.shape", output.shape)
        print("output", output)

        pred = torch.argmax(output).item()
        print("pred", pred)

        return {
            "asc_task_id": asc_task_id,
            "output": config.get_class_by_index(pred),
            "output_score": output[0][pred].item(),
        }


class AscTaskResource(Resource):

    def __init__(self):
        super(AscTaskResource, self).__init__()

    # @marshal_with(task_fields, envelope="task")
    def get(self, asc_task_id:int):

        # task = Task.query.get_or_404(task_id)

        return {"msg": "hi"} #task_schema.dump(task)