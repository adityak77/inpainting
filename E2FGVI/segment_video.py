import argparse
import os
import cv2
import tqdm
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 for segmentation masks on each frame")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path toresults/44403_results.mp4. config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def load_model(cfg):
    predictor = DefaultPredictor(cfg)
    return predictor

def process_video(model, video, downsample=None, resolution=None):
    count = 0
    while video.isOpened():
        success, frame = video.read()
        if success:
            count += 1
            if downsample is not None and count % downsample != 0:
                continue
            if resolution:
                frame = cv2.resize(frame, dsize=resolution, interpolation=cv2.INTER_CUBIC)
            yield (frame, model(frame))
        else:
            break

def get_segmented_frames(args, video_input, output=None, human_filter=False):
    cfg = setup_cfg(args)
    model = load_model(cfg)

    video = cv2.VideoCapture(video_input)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    downsample = num_frames // 60 + 1
    downsample_resolution_factor = max(width // 400 + 1, height // 400 + 1)
    resolution = width // downsample_resolution_factor, height // downsample_resolution_factor
    
    assert os.path.isfile(video_input)
    if output:
        assert os.path.isdir(output)

    frames_info = []
    frames = []
    for i, (frame, model_output) in enumerate(tqdm.tqdm(process_video(model, video, downsample=downsample, resolution=resolution), total=num_frames // downsample)):
        if output:
            save_path = os.path.join(output, f'frame{i}.png')
            save_path_original = os.path.join(output, f'original_frame{i}.png')
            
            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            vis_frame = v.draw_instance_predictions(model_output["instances"].to("cpu"))
            out = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(save_path, out)
            cv2.imwrite(save_path_original, frame) 

        frames_info.append(model_output)
        frames.append(frame)

    masks = []
    for i in range(len(frames_info)):
        if not human_filter:
            masks.append(frames_info[i]['instances'].pred_masks)
        else:
            human_masks = []
            for j, cls_id in enumerate(frames_info[i]['instances'].pred_classes):
                if cls_id == 0:
                    hmask = frames_info[i]['instances'].pred_masks[j]
                    struct_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                    hmask_processed = cv2.dilate(hmask.cpu().numpy().astype(np.uint8), struct_element, iterations=4).astype(np.bool)
                    hmask_processed = torch.Tensor(hmask_processed).to(hmask.device)
                    human_masks.append(hmask_processed)

            if len(human_masks) == 0:
                masks.append(torch.Tensor(human_masks))
            else:
                masks.append(torch.stack(human_masks))
    
    return frames, masks

if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    get_segmented_frames(args, args.video_input, output=args.output, human_filter=True)
