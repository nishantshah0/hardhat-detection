import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("universe-datasets").project("hard-hat-universe-0dy7t")
dataset = project.version(17).download("yolov11")
print("DATA IS AT:", dataset.location)