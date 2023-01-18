# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

import os
import cv2
import json
import numpy as np
import random

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

data_dir = 'robot_data_shadow'

def get_dicts(img_dir):
    anns = sorted([fname for fname in os.listdir(img_dir) if fname.endswith('.json')])
    imgs = sorted([fname for fname in os.listdir(img_dir) if fname.endswith('.png')])

    dataset_dicts = []

    for idx, (img_path, ann_path) in enumerate(zip(imgs, anns)):
        assert img_path.split('.')[0] == ann_path.split('.')[0], img_path + ' ' + ann_path

        img_full_path = os.path.join(img_dir, img_path)
        ann_full_path = os.path.join(img_dir, ann_path)
        with open(ann_full_path) as f:
            ann = json.load(f)

        record = {}
        
        record["file_name"] = img_full_path
        record["image_id"] = int(img_path.split('.')[0])
        record["height"] = ann['imageHeight']
        record["width"] = ann['imageWidth']
      
        objs = []
        obj = {}
        for shape in ann['shapes']:
            px, py = zip(*shape['points'])
            poly = [(np.floor(x) + 0.5, np.floor(y) + 0.5) for x, y in shape['points']]
            poly = [p for x in poly for p in x]

            if "bbox" not in obj:
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
            else:
                obj["bbox"] = [
                                np.min([np.min(px), obj["bbox"][0]]), 
                                np.min([np.min(py), obj["bbox"][1]]),
                                np.max([np.max(px), obj["bbox"][2]]),
                                np.max([np.max(py), obj["bbox"][3]])
                              ]
                obj["segmentation"] += [poly]
        
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("robot_" + d, lambda d=d: get_dicts(f"{data_dir}/" + d))
    MetadataCatalog.get("robot_" + d).set(thing_classes=["robot"])
robot_metadata = MetadataCatalog.get("robot_train")

dataset_dicts = get_dicts(f"{data_dir}/train")
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=robot_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite(f'{data_dir}/sample_mask.png', out.get_image()[:, :, ::-1])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("robot_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 1000 iterations seems good enough for this dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (robot). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


# Evaluation

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

dataset_dicts = get_dicts(f"{data_dir}/val")
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=robot_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(f'{data_dir}/inference.png', out.get_image()[:, :, ::-1])

evaluator = COCOEvaluator("robot_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "robot_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
