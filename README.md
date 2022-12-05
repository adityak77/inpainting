# inpainting

Various segmentation and video inpainting approaches with the objective to use it for reward learning.

## Setup

Three main environments: `detectron2`, `open-mmlab` and `e2fgvi`.

For the `detectron2` environment, do

```
cd detectron2
python -m pip install -e detectron2
```

For the `e2fgvi` environment, we need a specific version of detectron2 because e2fgvi uses torch 1.5.
```
cd E2FGVI
conda env create -f e2fgvi_detectron2.yml 
python -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

The `open-mmlab` environment is made for doing experiments with `EgoHOS` instead of `detectron`. Activate it as follows:
```
cd EgoHOS
conda env create -n open-mmlab python=3.7
pip install -r requirements.txt
pip install -U openmim
mim install mmcv-full==1.6.0
cd mmsegmentation
pip install -v -e .
```

In order to make this work in the same script, I have a mega environment that is compatible with all packages:
```
conda env create -f env_dvd_e2fgvi_detectron_egohos.yml
conda activate dvd_e2fgvi_detectron_egohos
```

## Detectron2 inference

To get a video visualization of the masks on an image, use the following:

```
conda activate detectron2
cd detectron2/demo

python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --video-input /path/to/video --output ../examples --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

The specification of the model weights as `detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl` indicates where the model should be downloaded from online.

To see individual frames output from a video, we can use a separate script here:

```
cd detectron2

python segment_video.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --video-input /path/to/video --output examples --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

## Training and building custom segmentation models

To train a model for segmenting robots (or for any custom dataset with LabelMe annotations), use the following script assuming that the robot data is in the `robot_data/` directory:

```
python train_segment_robot.py
```

The weights of the model will be stored in `output/model_final.pth`. To run inference using the model we trained above, we can modify our inference command as follows. Note that we modify some model options because the custom model has only one class (`MODEL.ROI_HEADS.NUM_CLASSES = 1`, `MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7`)

```
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --video-input /path/to/video --output ../examples --opts MODEL.WEIGHTS ../output/model_final.pth MODEL.ROI_HEADS.NUM_CLASSES 1 MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.7
```

## E2FGVI Video Inpainting

Specify options for detectron2 and it can be used with E2FGVI to output inpainted human videos in one script.

```
conda activate e2fgvi
cd E2FGVI

python test.py --model e2fgvi --video /path/to/video/ --neighbor_stride 1 --ckpt release_model/E2FGVI-CVPR22.pth --config-file ~/inpainting/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --opts [Model modifications here]
```

## Inference and Inpainting with EgoHOS

Inference to visualize segmentation masks: 
```
conda activate dvd_e2fgvi_detectron_egohos
cd EgoHOS/mmsegmentation
python segment_video_hands.py --video /path/to/video --output_file /path/to/output
```

Inpainting with EgoHOS' segmentation masks:
```
conda activate dvd_e2fgvi_detectron_egohos
cd E2FGVI
python test_egohos.py --model e2fgvi_hq --video /path/to/video/ --neighbor_stride 1 --ckpt release_model/E2FGVI-HQ-CVPR22.pth
```

## Stable Diffusion Video Inpainting

Specify options for detectron2 and it can be used with Stable Diffusion to output inpainted human videos in one script.

```
conda activate ldm_new
cd stable-diffusion
STEPS=50 # default but can be reduced to 10 probably

python scripts/inpaint_detectron.py --config-file ~/inpainting/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --video-input /path/to/video --outdir inpaint_examples --steps STEPS --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
