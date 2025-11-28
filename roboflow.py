from roboflow import Roboflow
rf = Roboflow(api_key="hoLO27xIY9UMRlUqmPbY")
project = rf.workspace("train-images-e1umq").project("facedetection-v5te5")
version = project.version(2)
dataset = version.download("yolov5")
