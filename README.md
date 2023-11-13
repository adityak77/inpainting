# Inpainting

Various segmentation and video inpainting approaches with the objective to use it for reward learning.

## Model downloads

- Copy and Paste Networks
```
cd Copy-and-Paste-Networks-for-Deep-Video-Inpainting/
mkdir ./weight
wget -O ./weight/weight.pth "https://www.dropbox.com/s/vbh12ay2ubrw3m9/weight.pth?dl=0"
```

- EgoHOS
```
pip install gdown
cd EgoHOS/mmsegmentation
gdown https://drive.google.com/uc?id=1LNMQ6TGf1QaCjMgTExPzl7lFFs-yZyqX
unzip work_dirs.zip
rm work_dirs.zip
```

- E2FGVI
```
cd E2FGVI/release_model
gdown 10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3 # E2FGVI-HQ
```

- Detectron2
In order to get trained robot segmentation model, run (should take less than a minute)
```
cd detectron2
conda activate detectron2 # see instructions below
python train_segment_robot.py
```

- Stable Diffusion: get models from huggingface

## Setup

Three main environments: `detectron2`, `open-mmlab` and `e2fgvi`.

For the `detectron2` environment, do

```
cd detectron2
conda create --name detectron2 python=3.7
conda activate detectron2
conda install pytorch=1.10.0 torchvision cudatoolkit=10.2 -c pytorch
python -m pip install -e .
pip install setuptools==59.5.0
```

For the `e2fgvi` environment, we need a specific version of detectron2.
```
cd E2FGVI
conda env create -f e2fgvi_detectron2.yml 
python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
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
python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

cd ~/rewards-from-human-videos/dvd/sim_envs
pip install -e .

cd ~/rewards-from-human-videos/metaworld
pip install -e .

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

## Textual Inversion training and inference

For training, run the following. Running on one GPU with 256x256 resolution runs in around 30 min.
```
cd diffusers
pip install -e .

cd examples/textual_inversion
pip install -r requirements.txt
accelerate config

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR=/home/akannan2/inpainting/stable-diffusion/robot_style
export OUTPUT="textual_inversion_sim_style"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<sim-style>" --initializer_token="animation" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=OUTPUT
```

This outputs the model in `OUTPUT`. To convert it to a .ckpt file, run
```
mkdir models/ldm/stable-diffusion-v1-text-inversion/

python ~/inpainting/diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py \
--model_path textual_inversion_sim_style \
--checkpoint_path ~/inpainting/stable-diffusion/models/ldm/stable-diffusion-v1-text-inversion/model.ckpt
```

For inference,
```
python scripts/img2img.py --prompt "stapler, in style of <sim-style>" \
--init-img /path/to/img/ \
--strength 0.5 --n_samples 4 \
--ckpt models/ldm/stable-diffusion-v1-text-inversion/model.ckpt
```

