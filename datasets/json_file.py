import json

json_path = "../datasets/coco2017/annotations/instances_train2017.json"
json_labels = json.load(open(json_path, "r"))
print(json_labels["info"])
